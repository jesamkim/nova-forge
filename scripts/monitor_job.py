#!/usr/bin/env python3
"""
monitor_job.py - Monitor the status of a Bedrock model customization (fine-tuning) job.

Usage:
    # Show latest job
    python scripts/monitor_job.py

    # Show a specific job by ARN
    python scripts/monitor_job.py --job-arn arn:aws:bedrock:...

    # Watch mode: poll every 60 seconds until completion or Ctrl-C
    python scripts/monitor_job.py --watch
    python scripts/monitor_job.py --watch --job-arn arn:aws:bedrock:...
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AWS_PROFILE = "profile2"
AWS_REGION = "us-east-1"
WATCH_INTERVAL_SECONDS = 60

# Terminal states – stop polling when the job reaches one of these
TERMINAL_STATES = {"Completed", "Failed", "Stopped"}

# ANSI colour codes for readability (no-op on terminals that don't support them)
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_session() -> boto3.Session:
    """Create a boto3 session using the designated AWS profile."""
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def list_jobs(bedrock_client) -> list[dict]:
    """
    Return all model customization jobs, newest first.
    Handles pagination automatically.
    """
    jobs: list[dict] = []
    paginator = bedrock_client.get_paginator("list_model_customization_jobs")
    for page in paginator.paginate():
        jobs.extend(page.get("modelCustomizationJobSummaries", []))

    # Sort by creation time descending so jobs[0] is the latest
    jobs.sort(key=lambda j: j.get("creationTime", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return jobs


def get_job(bedrock_client, job_arn: str) -> dict:
    """Fetch full details of a single customization job."""
    return bedrock_client.get_model_customization_job(jobIdentifier=job_arn)


def resolve_job_arn(bedrock_client, job_arn: str | None) -> str:
    """
    Return *job_arn* as-is when provided, otherwise pick the most
    recently created job from the account.
    """
    if job_arn:
        return job_arn

    print("[INFO] No --job-arn provided; fetching the most recent job ...")
    jobs = list_jobs(bedrock_client)
    if not jobs:
        print("[ERROR] No model customization jobs found in this account.", file=sys.stderr)
        sys.exit(1)

    latest = jobs[0]
    arn = latest["jobArn"]
    print(f"[INFO] Latest job: {latest.get('jobName', '<unnamed>')}  ({arn})")
    return arn


def format_timestamp(dt: datetime | None) -> str:
    """Format a datetime object for display, or return '-' if None."""
    if dt is None:
        return "-"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def colorize_status(status: str) -> str:
    """Wrap a status string with an ANSI colour code."""
    if status == "Completed":
        return f"{_GREEN}{status}{_RESET}"
    if status == "Failed":
        return f"{_RED}{status}{_RESET}"
    if status in ("Stopping", "Stopped"):
        return f"{_YELLOW}{status}{_RESET}"
    # InProgress / Submitted / etc.
    return f"{_CYAN}{status}{_RESET}"


def print_job_summary(job: dict) -> None:
    """Print a human-readable summary of a customization job."""
    status = job.get("status", "Unknown")
    print("\n" + "=" * 60)
    print(f"  Job Name   : {job.get('jobName', '-')}")
    print(f"  Job ARN    : {job.get('jobArn', '-')}")
    print(f"  Base Model : {job.get('baseModelArn', '-')}")
    print(f"  Status     : {colorize_status(status)}")
    print(f"  Created    : {format_timestamp(job.get('creationTime'))}")
    print(f"  Last Upd.  : {format_timestamp(job.get('lastModifiedTime'))}")
    if job.get("endTime"):
        print(f"  End Time   : {format_timestamp(job.get('endTime'))}")

    # Show output model ARN when completed
    if status == "Completed":
        output_model = job.get("outputModelArn") or job.get("outputModelName", "-")
        print(f"  Output     : {_GREEN}{output_model}{_RESET}")

    # Show failure message when failed
    if status == "Failed":
        failure_msg = job.get("failureMessage", "No failure message available.")
        print(f"  {_RED}Failure Msg: {failure_msg}{_RESET}")

    # Training metrics if available
    metrics = job.get("trainingMetrics") or {}
    if metrics:
        print(f"  Train Loss : {metrics.get('trainingLoss', '-')}")

    validation_metrics = job.get("validationMetrics") or []
    if validation_metrics:
        for vm in validation_metrics:
            print(f"  Valid. Loss: {vm.get('validationLoss', '-')}")

    print("=" * 60 + "\n")


def print_all_jobs(jobs: list[dict]) -> None:
    """Print a compact table listing all customization jobs."""
    if not jobs:
        print("[INFO] No customization jobs found.")
        return

    header = f"{'#':<4} {'Status':<14} {'Job Name':<45} {'Created'}"
    print("\n" + header)
    print("-" * len(header))
    for idx, job in enumerate(jobs, start=1):
        status = job.get("status", "Unknown")
        name = job.get("jobName", "-")[:44]
        created = format_timestamp(job.get("creationTime"))
        status_col = colorize_status(status)
        # Pad without colour codes distorting alignment
        print(f"{idx:<4} {status_col:<23} {name:<45} {created}")
    print()


# ---------------------------------------------------------------------------
# Core monitor logic
# ---------------------------------------------------------------------------

def check_once(bedrock_client, job_arn: str) -> str:
    """
    Fetch and print the current status of *job_arn*.
    Returns the status string.
    """
    try:
        job = get_job(bedrock_client, job_arn)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        print(f"[ERROR] get_model_customization_job failed ({error_code}): {exc}", file=sys.stderr)
        sys.exit(1)

    print_job_summary(job)
    return job.get("status", "Unknown")


def monitor(job_arn: str | None, watch: bool, list_all: bool) -> None:
    """Entry point for the monitoring workflow."""
    session = build_session()
    bedrock = session.client("bedrock")

    # --list: show all jobs and exit
    if list_all:
        print("[INFO] Listing all model customization jobs ...")
        try:
            jobs = list_jobs(bedrock)
        except (BotoCoreError, ClientError) as exc:
            print(f"[ERROR] list_model_customization_jobs failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print_all_jobs(jobs)
        return

    # Resolve which job to inspect
    try:
        resolved_arn = resolve_job_arn(bedrock, job_arn)
    except (BotoCoreError, ClientError) as exc:
        print(f"[ERROR] Failed to list jobs: {exc}", file=sys.stderr)
        sys.exit(1)

    if not watch:
        # Single check
        check_once(bedrock, resolved_arn)
        return

    # Watch mode: poll at regular intervals
    print(f"[INFO] Watch mode enabled. Polling every {WATCH_INTERVAL_SECONDS}s. Press Ctrl-C to stop.\n")
    try:
        while True:
            now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[{now}] Checking job status ...")
            status = check_once(bedrock, resolved_arn)

            if status in TERMINAL_STATES:
                print(f"[INFO] Job reached terminal state '{status}'. Stopping watch.")
                break

            print(f"[INFO] Next check in {WATCH_INTERVAL_SECONDS}s ...")
            time.sleep(WATCH_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[INFO] Watch interrupted by user.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Amazon Bedrock model customization (fine-tuning) jobs."
    )
    parser.add_argument(
        "--job-arn",
        metavar="ARN",
        help="ARN of the customization job to inspect. "
             "If omitted, the most recently created job is used.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=f"Poll the job every {WATCH_INTERVAL_SECONDS}s until it reaches a terminal state.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_all",
        help="List all customization jobs in the account and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    monitor(job_arn=args.job_arn, watch=args.watch, list_all=args.list_all)
