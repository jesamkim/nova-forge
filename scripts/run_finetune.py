#!/usr/bin/env python3
"""
run_finetune.py - Submit a Nova Micro SFT fine-tuning job to Amazon Bedrock.

Usage:
    python scripts/run_finetune.py
    python scripts/run_finetune.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AWS_PROFILE = "profile2"
AWS_REGION = "us-east-1"
BASE_MODEL_ID = "amazon.nova-micro-v1:0:128k"
CUSTOMIZATION_TYPE = "FINE_TUNING"
ROLE_NAME = "NovaForgeExperimentRole"
CUSTOM_MODEL_NAME = "nova-micro-sentiment-kr"

HYPERPARAMETERS = {
    "epochCount": "3",
    "batchSize": "1",
    "learningRate": "0.00001",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_session() -> boto3.Session:
    """Create a boto3 session using the designated AWS profile."""
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def get_account_id(session: boto3.Session) -> str:
    """Retrieve the AWS account ID via STS."""
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    return identity["Account"]


def get_role_arn(session: boto3.Session, role_name: str) -> str:
    """Look up the ARN of an IAM role by name."""
    iam = session.client("iam")
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]


def build_job_params(
    job_name: str,
    bucket_name: str,
    role_arn: str,
) -> dict:
    """Assemble the full parameter dict for create_model_customization_job."""
    return {
        "jobName": job_name,
        "customModelName": CUSTOM_MODEL_NAME,
        "roleArn": role_arn,
        "baseModelIdentifier": BASE_MODEL_ID,
        "customizationType": CUSTOMIZATION_TYPE,
        "hyperParameters": HYPERPARAMETERS,
        "trainingDataConfig": {
            "s3Uri": f"s3://{bucket_name}/data/train.jsonl",
        },
        "validationDataConfig": {
            "validators": [
                {
                    "s3Uri": f"s3://{bucket_name}/data/val.jsonl",
                }
            ]
        },
        "outputDataConfig": {
            "s3Uri": f"s3://{bucket_name}/output/",
        },
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(dry_run: bool) -> None:
    """Submit (or preview) the fine-tuning job."""
    session = build_session()

    # Resolve account-specific resource names
    print("[INFO] Fetching AWS account ID ...")
    try:
        account_id = get_account_id(session)
    except (BotoCoreError, ClientError) as exc:
        print(f"[ERROR] Failed to retrieve account ID: {exc}", file=sys.stderr)
        sys.exit(1)

    bucket_name = f"nova-forge-experiment-{account_id}"
    print(f"[INFO] Account ID  : {account_id}")
    print(f"[INFO] S3 Bucket   : s3://{bucket_name}")

    # Resolve IAM role ARN
    print(f"[INFO] Fetching ARN for IAM role '{ROLE_NAME}' ...")
    try:
        role_arn = get_role_arn(session, ROLE_NAME)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "NoSuchEntity":
            print(
                f"[ERROR] IAM role '{ROLE_NAME}' not found. "
                "Run setup_infra.sh first.",
                file=sys.stderr,
            )
        else:
            print(f"[ERROR] IAM error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Role ARN    : {role_arn}")

    # Build job name with UTC timestamp for uniqueness
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"nova-micro-sentiment-{timestamp}"

    params = build_job_params(
        job_name=job_name,
        bucket_name=bucket_name,
        role_arn=role_arn,
    )

    # Preview mode: print params and exit without submitting
    if dry_run:
        print("\n[DRY RUN] Job parameters (not submitted):")
        print(json.dumps(params, indent=2))
        return

    # Submit the fine-tuning job
    bedrock = session.client("bedrock")
    print(f"\n[INFO] Submitting fine-tuning job: {job_name}")
    try:
        response = bedrock.create_model_customization_job(**params)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        print(f"[ERROR] create_model_customization_job failed ({error_code}): {exc}", file=sys.stderr)
        sys.exit(1)

    job_arn = response.get("jobArn", "<unknown>")
    print("\n[OK] Fine-tuning job submitted successfully.")
    print(f"     Job Name : {job_name}")
    print(f"     Job ARN  : {job_arn}")
    print(f"\nMonitor with:")
    print(f"     python scripts/monitor_job.py --job-arn {job_arn}")
    print(f"     python scripts/monitor_job.py --watch --job-arn {job_arn}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a Nova Micro SFT fine-tuning job to Amazon Bedrock."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job parameters without actually submitting the job.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run)
