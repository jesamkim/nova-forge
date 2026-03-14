#!/bin/bash
# =============================================================================
# setup_infra.sh - Nova Forge 실험용 AWS 인프라 설정
#
# 사용법:
#   ./scripts/setup_infra.sh
#
# 생성 리소스:
#   - S3 버킷: nova-forge-experiment-{account_id}
#   - IAM Role: NovaForgeExperimentRole (Bedrock trust policy)
#   - IAM Policy: NovaForgeExperimentPolicy (S3 + Bedrock + CloudWatch)
#
# 멱등성: 이미 존재하는 리소스는 건너뜁니다 (재실행 안전).
# =============================================================================
set -euo pipefail

export AWS_PROFILE=profile2
REGION="us-east-1"
ROLE_NAME="NovaForgeExperimentRole"
POLICY_NAME="NovaForgeExperimentPolicy"

# ── 계정 ID 동적 조회 ──────────────────────────────────────────────────────────
echo "[INFO] AWS 계정 ID 조회 중..."
ACCOUNT_ID=$(aws sts get-caller-identity \
  --profile profile2 \
  --query Account \
  --output text)
echo "[INFO] Account ID: ${ACCOUNT_ID}"

BUCKET_NAME="nova-forge-experiment-${ACCOUNT_ID}"

# ── S3 버킷 생성 ───────────────────────────────────────────────────────────────
echo "[INFO] S3 버킷 확인 중: ${BUCKET_NAME}"
if aws s3api head-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" 2>/dev/null; then
  echo "[SKIP] S3 버킷이 이미 존재합니다: ${BUCKET_NAME}"
else
  echo "[INFO] S3 버킷 생성 중: ${BUCKET_NAME}"
  # us-east-1은 CreateBucketConfiguration 없이 생성
  aws s3api create-bucket \
    --bucket "${BUCKET_NAME}" \
    --region "${REGION}" \
    --profile profile2

  # 퍼블릭 액세스 차단
  aws s3api put-public-access-block \
    --bucket "${BUCKET_NAME}" \
    --public-access-block-configuration \
      "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" \
    --profile profile2

  # 버킷 버저닝 활성화
  aws s3api put-bucket-versioning \
    --bucket "${BUCKET_NAME}" \
    --versioning-configuration Status=Enabled \
    --profile profile2

  echo "[OK] S3 버킷 생성 완료: s3://${BUCKET_NAME}"
fi

# ── S3 폴더 구조 생성 (placeholder 오브젝트) ────────────────────────────────────
echo "[INFO] S3 폴더 구조 초기화 중..."
for prefix in "data/" "output/"; do
  if aws s3api head-object --bucket "${BUCKET_NAME}" --key "${prefix}.keep" --profile profile2 2>/dev/null; then
    echo "[SKIP] 폴더 이미 존재: s3://${BUCKET_NAME}/${prefix}"
  else
    echo "" | aws s3 cp - "s3://${BUCKET_NAME}/${prefix}.keep" --profile profile2
    echo "[OK] 폴더 생성: s3://${BUCKET_NAME}/${prefix}"
  fi
done

# ── IAM Trust Policy 정의 ──────────────────────────────────────────────────────
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "aws:SourceAccount": "${ACCOUNT_ID}"
        },
        "ArnLike": {
          "aws:SourceArn": "arn:aws:bedrock:${REGION}:${ACCOUNT_ID}:model-customization-job/*"
        }
      }
    }
  ]
}
EOF
)

# ── IAM Permission Policy 정의 ────────────────────────────────────────────────
PERMISSION_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3TrainingDataAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Sid": "BedrockModelAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateModelCustomizationJob",
        "bedrock:GetModelCustomizationJob",
        "bedrock:ListModelCustomizationJobs",
        "bedrock:StopModelCustomizationJob",
        "bedrock:GetCustomModel",
        "bedrock:ListCustomModels",
        "bedrock:DeleteCustomModel",
        "bedrock:CreateProvisionedModelThroughput",
        "bedrock:GetProvisionedModelThroughput",
        "bedrock:DeleteProvisionedModelThroughput",
        "bedrock:InvokeModel"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogsAccess",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock/*"
    }
  ]
}
EOF
)

# ── IAM Role 생성 ──────────────────────────────────────────────────────────────
echo "[INFO] IAM Role 확인 중: ${ROLE_NAME}"
if aws iam get-role --role-name "${ROLE_NAME}" --profile profile2 2>/dev/null; then
  echo "[SKIP] IAM Role이 이미 존재합니다: ${ROLE_NAME}"
else
  echo "[INFO] IAM Role 생성 중: ${ROLE_NAME}"
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" \
    --description "Nova Forge fine-tuning experiment Bedrock service role" \
    --profile profile2
  echo "[OK] IAM Role 생성 완료: ${ROLE_NAME}"
fi

# ── IAM Policy 생성 또는 업데이트 ─────────────────────────────────────────────
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
echo "[INFO] IAM Policy 확인 중: ${POLICY_NAME}"
if aws iam get-policy --policy-arn "${POLICY_ARN}" --profile profile2 2>/dev/null; then
  echo "[SKIP] IAM Policy가 이미 존재합니다: ${POLICY_ARN}"
else
  echo "[INFO] IAM Policy 생성 중: ${POLICY_NAME}"
  aws iam create-policy \
    --policy-name "${POLICY_NAME}" \
    --policy-document "${PERMISSION_POLICY}" \
    --description "Nova Forge fine-tuning experiment least-privilege policy" \
    --profile profile2
  echo "[OK] IAM Policy 생성 완료: ${POLICY_ARN}"
fi

# ── Policy를 Role에 Attach ────────────────────────────────────────────────────
echo "[INFO] Policy를 Role에 연결 중..."
ATTACHED=$(aws iam list-attached-role-policies \
  --role-name "${ROLE_NAME}" \
  --profile profile2 \
  --query "AttachedPolicies[?PolicyArn=='${POLICY_ARN}'].PolicyArn" \
  --output text)

if [[ -n "${ATTACHED}" ]]; then
  echo "[SKIP] Policy가 이미 Role에 연결되어 있습니다."
else
  aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn "${POLICY_ARN}" \
    --profile profile2
  echo "[OK] Policy 연결 완료."
fi

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
ROLE_ARN=$(aws iam get-role \
  --role-name "${ROLE_NAME}" \
  --profile profile2 \
  --query Role.Arn \
  --output text)

echo ""
echo "============================================================"
echo "  Nova Forge 인프라 설정 완료"
echo "============================================================"
echo "  S3 Bucket : s3://${BUCKET_NAME}"
echo "  IAM Role  : ${ROLE_ARN}"
echo "  Policy    : ${POLICY_ARN}"
echo "  Region    : ${REGION}"
echo "============================================================"
echo ""
echo "다음 단계: ./scripts/upload_data.sh 로 학습 데이터 업로드"
