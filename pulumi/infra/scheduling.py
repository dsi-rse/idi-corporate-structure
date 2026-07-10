"""EventBridge Scheduler for triggering ECS Fargate tasks on a cron schedule.

Includes:
  - SQS dead-letter queue for failed invocations
  - IAM role for the scheduler to invoke ecs:RunTask
  - Schedule with configurable cron expression (starts disabled)
"""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecs, iam, networking

# -----------------------------------------------------------------------------
# Shared DLQ (looked up by name — owned by a separate project/stack)
# Name comes from SSM (/idi/<stack>/shared/dlq_name) via config.
# -----------------------------------------------------------------------------
shared_dlq = aws.sqs.get_queue_output(name=config.shared_dlq_name)

# -----------------------------------------------------------------------------
# Scheduler IAM Role — allows EventBridge to run ECS tasks and send to DLQ
# -----------------------------------------------------------------------------
scheduler_role = aws.iam.Role(
    "idi-role-scheduler",
    name=f"{config.name_prefix}-role-scheduler",
    description="EventBridge Scheduler role: run ECS tasks, pass roles, send to DLQ",
    assume_role_policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "scheduler.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    ),
    tags=config.tags(),
)

scheduler_policy = aws.iam.RolePolicy(
    "idi-policy-scheduler",
    role=scheduler_role.id,
    policy=pulumi.Output.all(
        task_execution_role_arn=iam.task_execution_role.arn,
        task_role_arn=iam.task_role.arn,
        dlq_arn=shared_dlq.arn,
        task_definition_arn=ecs.task_definition.arn,
    ).apply(
        lambda args: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "ecs:RunTask",
                        "Resource": args["task_definition_arn"],
                    },
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": [
                            args["task_execution_role_arn"],
                            args["task_role_arn"],
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": "sqs:SendMessage",
                        "Resource": args["dlq_arn"],
                    },
                ],
            }
        )
    ),
)


# -----------------------------------------------------------------------------
# EventBridge Schedule
# -----------------------------------------------------------------------------
schedule_expression = config.config.get("cron_corporate_structure") or "cron(0 2 * * ? *)"
schedule_enabled = (config.config.get("schedule_enabled") or "false") == "true"

_CONTAINER_NAME = ecs.CONTAINER_NAME


def _container_override_input() -> str:
    """Build the EventBridge Scheduler `input` JSON for an ECS containerOverride.

    Mirrors the command in the task definition so the schedule drives execution
    with the same arguments. All values resolve to plain strings at plan time.

    ``--daily`` processes the most recent filings (default 7-day look-back) from
    the SEC bucket's manifest; the OpenAI key and SEC User-Agent are supplied via
    the task's environment (see ecs.py), not on the command line.
    """
    command = [
        "--sec-bucket-prefix",
        ecs.sec_bucket_prefix,
        "--output-file",
        ecs.output_file,
        "--failure-file",
        ecs.failure_file,
        "--daily",
        "--rate-limit",
        ecs.rate_limit,
        "--num-workers",
        ecs.num_workers,
        "--model",
        ecs.openai_model,
    ]
    return json.dumps({"containerOverrides": [{"name": _CONTAINER_NAME, "command": command}]})


schedule = aws.scheduler.Schedule(
    "idi-schedule-corporate-structure",
    name=f"{config.name_prefix}-schedule",
    description="Triggers the corporate structure processor ECS task",
    schedule_expression=schedule_expression,
    flexible_time_window=aws.scheduler.ScheduleFlexibleTimeWindowArgs(
        mode="OFF",
    ),
    state="ENABLED" if schedule_enabled else "DISABLED",
    target=aws.scheduler.ScheduleTargetArgs(
        arn=ecs.cluster.arn,
        role_arn=scheduler_role.arn,
        input=_container_override_input(),
        ecs_parameters=aws.scheduler.ScheduleTargetEcsParametersArgs(
            task_definition_arn=ecs.task_definition.arn,
            launch_type="FARGATE",
            platform_version="LATEST",
            enable_execute_command=True,
            propagate_tags="TASK_DEFINITION",
            network_configuration=aws.scheduler.ScheduleTargetEcsParametersNetworkConfigurationArgs(
                assign_public_ip=True,
                subnets=[networking.primary_subnet_id],
                security_groups=[networking.ecs_sg.id],
            ),
        ),
        retry_policy=aws.scheduler.ScheduleTargetRetryPolicyArgs(
            maximum_retry_attempts=2,
            maximum_event_age_in_seconds=3600,
        ),
        dead_letter_config=aws.scheduler.ScheduleTargetDeadLetterConfigArgs(
            arn=shared_dlq.arn,
        ),
    ),
)
