#!/bin/bash
# =============================================================================
# cleanup_infra.sh - Nova Forge 실험 리소스 정리
#
# 사용법:
#   ./scripts/cleanup_infra.sh             # 확인 프롬프트 후 삭제
#   ./scripts/cleanup_infra.sh --force     # 확인 없이 즉시 삭제 (CI/CD 전용)
#   ./scripts/cleanup_infra.sh --dry-run   # 삭제 대상 목록만 출력
#
# 삭제 대상:
#   1. S3 버킷 내 모든 오브젝트 및 버전 (버킷 포함)
#   2. IAM Policy (Role에서 detach 후 삭제)
#   3. IAM Role (NovaForgeExperimentRole)
#
# 주의: 이 작업은 되돌릴 수 없습니다.
# =============================================================================
set -euo pipefail

export AWS_PROFILE=profile2
REGION="us-east-1"
ROLE_NAME="NovaForgeExperimentRole"
POLICY_NAME="NovaForgeExperimentPolicy"

# ── 옵션 파싱 ─────────────────────────────────────────────────────────────────
FORCE=false
DRY_RUN=false
for arg in "$@"; do
  case "${arg}" in
    --force)   FORCE=true ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "[ERROR] 알 수 없는 옵션: ${arg}"; exit 1 ;;
  esac
done

# ── 계정 ID 조회 ───────────────────────────────────────────────────────────────
echo "[INFO] AWS 계정 ID 조회 중..."
ACCOUNT_ID=$(aws sts get-caller-identity \
  --profile profile2 \
  --query Account \
  --output text)
echo "[INFO] Account ID: ${ACCOUNT_ID}"

BUCKET_NAME="nova-forge-experiment-${ACCOUNT_ID}"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"

# ── 삭제 대상 목록 출력 ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  삭제 예정 리소스"
echo "============================================================"
echo "  S3 Bucket : s3://${BUCKET_NAME} (모든 오브젝트 포함)"
echo "  IAM Role  : ${ROLE_NAME}"
echo "  IAM Policy: ${POLICY_ARN}"
echo "  Region    : ${REGION}"
echo "============================================================"
echo ""

# ── dry-run 모드 ──────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY-RUN] S3 버킷 내 오브젝트 목록:"
  if aws s3api head-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" 2>/dev/null; then
    aws s3 ls "s3://${BUCKET_NAME}/" --recursive --profile profile2 || true
  else
    echo "  (버킷이 존재하지 않음)"
  fi
  echo "[DRY-RUN] 실제 삭제는 수행하지 않았습니다."
  exit 0
fi

# ── 확인 프롬프트 ─────────────────────────────────────────────────────────────
if [[ "${FORCE}" == "false" ]]; then
  echo "경고: 위 리소스가 영구적으로 삭제됩니다."
  echo -n "계속하려면 'DELETE'를 입력하세요: "
  read -r confirmation
  if [[ "${confirmation}" != "DELETE" ]]; then
    echo "[ABORT] 삭제가 취소되었습니다."
    exit 0
  fi
fi

# ── S3 버킷 삭제 ──────────────────────────────────────────────────────────────
echo "[INFO] S3 버킷 삭제 중: ${BUCKET_NAME}"
if ! aws s3api head-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" 2>/dev/null; then
  echo "[SKIP] S3 버킷이 존재하지 않습니다: ${BUCKET_NAME}"
else
  # 버킷 버저닝 활성화 상태이면 모든 버전 및 삭제 마커 제거 필요
  echo "[INFO] 오브젝트 버전 전체 삭제 중 (버저닝 포함)..."

  # 현재 오브젝트 삭제
  aws s3 rm "s3://${BUCKET_NAME}" --recursive --profile profile2 || true

  # 버전 관리된 오브젝트 삭제
  VERSION_IDS=$(aws s3api list-object-versions \
    --bucket "${BUCKET_NAME}" \
    --profile profile2 \
    --query 'Versions[].{Key:Key,VersionId:VersionId}' \
    --output json 2>/dev/null || echo "[]")

  if [[ "${VERSION_IDS}" != "[]" && -n "${VERSION_IDS}" ]]; then
    echo "[INFO] 이전 버전 오브젝트 삭제 중..."
    echo "${VERSION_IDS}" | python3 -c "
import sys, json, subprocess
versions = json.load(sys.stdin)
if not versions:
    sys.exit(0)
objects = [{'Key': v['Key'], 'VersionId': v['VersionId']} for v in versions]
delete_payload = json.dumps({'Objects': objects, 'Quiet': True})
subprocess.run([
    'aws', 's3api', 'delete-objects',
    '--bucket', '${BUCKET_NAME}',
    '--delete', delete_payload,
    '--profile', 'profile2'
], check=True)
print(f'[OK] {len(objects)}개 버전 오브젝트 삭제 완료')
"
  fi

  # 삭제 마커 제거
  DELETE_MARKERS=$(aws s3api list-object-versions \
    --bucket "${BUCKET_NAME}" \
    --profile profile2 \
    --query 'DeleteMarkers[].{Key:Key,VersionId:VersionId}' \
    --output json 2>/dev/null || echo "[]")

  if [[ "${DELETE_MARKERS}" != "[]" && -n "${DELETE_MARKERS}" ]]; then
    echo "[INFO] 삭제 마커 제거 중..."
    echo "${DELETE_MARKERS}" | python3 -c "
import sys, json, subprocess
markers = json.load(sys.stdin)
if not markers:
    sys.exit(0)
objects = [{'Key': m['Key'], 'VersionId': m['VersionId']} for m in markers]
delete_payload = json.dumps({'Objects': objects, 'Quiet': True})
subprocess.run([
    'aws', 's3api', 'delete-objects',
    '--bucket', '${BUCKET_NAME}',
    '--delete', delete_payload,
    '--profile', 'profile2'
], check=True)
print(f'[OK] {len(objects)}개 삭제 마커 제거 완료')
"
  fi

  # 버킷 삭제
  aws s3api delete-bucket \
    --bucket "${BUCKET_NAME}" \
    --region "${REGION}" \
    --profile profile2
  echo "[OK] S3 버킷 삭제 완료: ${BUCKET_NAME}"
fi

# ── IAM Policy Detach 및 삭제 ────────────────────────────────────────────────
echo "[INFO] IAM Policy 처리 중: ${POLICY_NAME}"

# Role에서 Policy detach
if aws iam get-role --role-name "${ROLE_NAME}" --profile profile2 2>/dev/null; then
  ATTACHED=$(aws iam list-attached-role-policies \
    --role-name "${ROLE_NAME}" \
    --profile profile2 \
    --query "AttachedPolicies[?PolicyArn=='${POLICY_ARN}'].PolicyArn" \
    --output text 2>/dev/null || true)

  if [[ -n "${ATTACHED}" ]]; then
    echo "[INFO] Role에서 Policy detach 중..."
    aws iam detach-role-policy \
      --role-name "${ROLE_NAME}" \
      --policy-arn "${POLICY_ARN}" \
      --profile profile2
    echo "[OK] Policy detach 완료."
  fi
fi

# Policy 삭제 (모든 non-default 버전 먼저 삭제)
if aws iam get-policy --policy-arn "${POLICY_ARN}" --profile profile2 2>/dev/null; then
  # 비기본 버전 삭제
  NON_DEFAULT_VERSIONS=$(aws iam list-policy-versions \
    --policy-arn "${POLICY_ARN}" \
    --profile profile2 \
    --query 'Versions[?IsDefaultVersion==`false`].VersionId' \
    --output text 2>/dev/null || true)

  for version_id in ${NON_DEFAULT_VERSIONS}; do
    echo "[INFO] Policy 버전 삭제: ${version_id}"
    aws iam delete-policy-version \
      --policy-arn "${POLICY_ARN}" \
      --version-id "${version_id}" \
      --profile profile2
  done

  aws iam delete-policy \
    --policy-arn "${POLICY_ARN}" \
    --profile profile2
  echo "[OK] IAM Policy 삭제 완료: ${POLICY_ARN}"
else
  echo "[SKIP] IAM Policy가 존재하지 않습니다: ${POLICY_ARN}"
fi

# ── IAM Role 삭제 ──────────────────────────────────────────────────────────────
echo "[INFO] IAM Role 삭제 중: ${ROLE_NAME}"
if ! aws iam get-role --role-name "${ROLE_NAME}" --profile profile2 2>/dev/null; then
  echo "[SKIP] IAM Role이 존재하지 않습니다: ${ROLE_NAME}"
else
  # 남은 inline policy 삭제
  INLINE_POLICIES=$(aws iam list-role-policies \
    --role-name "${ROLE_NAME}" \
    --profile profile2 \
    --query 'PolicyNames' \
    --output text 2>/dev/null || true)

  for policy_name in ${INLINE_POLICIES}; do
    echo "[INFO] Inline policy 삭제: ${policy_name}"
    aws iam delete-role-policy \
      --role-name "${ROLE_NAME}" \
      --policy-name "${policy_name}" \
      --profile profile2
  done

  aws iam delete-role \
    --role-name "${ROLE_NAME}" \
    --profile profile2
  echo "[OK] IAM Role 삭제 완료: ${ROLE_NAME}"
fi

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Nova Forge 인프라 정리 완료"
echo "============================================================"
echo "  삭제됨: s3://${BUCKET_NAME}"
echo "  삭제됨: IAM Role ${ROLE_NAME}"
echo "  삭제됨: IAM Policy ${POLICY_NAME}"
echo "============================================================"
