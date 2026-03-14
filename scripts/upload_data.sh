#!/bin/bash
# =============================================================================
# upload_data.sh - Nova Forge 학습 데이터 S3 업로드
#
# 사용법:
#   ./scripts/upload_data.sh
#   ./scripts/upload_data.sh --dry-run   # 실제 업로드 없이 대상 파일 목록만 출력
#
# 전제 조건:
#   - setup_infra.sh 실행 완료 (S3 버킷 존재해야 함)
#   - 로컬 data/ 디렉토리에 train.jsonl, val.jsonl 존재
#
# 업로드 대상:
#   로컬: data/train.jsonl  →  s3://{bucket}/data/train.jsonl
#   로컬: data/val.jsonl    →  s3://{bucket}/data/val.jsonl
# =============================================================================
set -euo pipefail

export AWS_PROFILE=profile2
REGION="us-east-1"

# ── 옵션 파싱 ─────────────────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    *) echo "[ERROR] 알 수 없는 옵션: ${arg}"; exit 1 ;;
  esac
done

# ── 스크립트 위치 기준으로 프로젝트 루트 탐색 ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_DIR="${PROJECT_ROOT}/data"

# ── 계정 ID 조회 ───────────────────────────────────────────────────────────────
echo "[INFO] AWS 계정 ID 조회 중..."
ACCOUNT_ID=$(aws sts get-caller-identity \
  --profile profile2 \
  --query Account \
  --output text)
echo "[INFO] Account ID: ${ACCOUNT_ID}"

BUCKET_NAME="nova-forge-experiment-${ACCOUNT_ID}"

# ── S3 버킷 존재 여부 확인 ────────────────────────────────────────────────────
echo "[INFO] S3 버킷 확인 중: ${BUCKET_NAME}"
if ! aws s3api head-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" 2>/dev/null; then
  echo "[ERROR] S3 버킷이 존재하지 않습니다: ${BUCKET_NAME}"
  echo "        먼저 ./scripts/setup_infra.sh 를 실행하세요."
  exit 1
fi

# ── 로컬 data/ 디렉토리 확인 ──────────────────────────────────────────────────
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] 로컬 data/ 디렉토리가 없습니다: ${DATA_DIR}"
  exit 1
fi

# ── 업로드 대상 파일 정의 ────────────────────────────────────────────────────
declare -A UPLOAD_MAP=(
  ["${DATA_DIR}/train.jsonl"]="data/train.jsonl"
  ["${DATA_DIR}/val.jsonl"]="data/val.jsonl"
)

# ── 파일 존재 여부 사전 검사 ──────────────────────────────────────────────────
MISSING=()
for local_path in "${!UPLOAD_MAP[@]}"; do
  if [[ ! -f "${local_path}" ]]; then
    MISSING+=("${local_path}")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[ERROR] 다음 파일이 없습니다:"
  for f in "${MISSING[@]}"; do
    echo "        - ${f}"
  done
  exit 1
fi

# ── dry-run 모드 ──────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY-RUN] 업로드 예정 파일:"
  for local_path in "${!UPLOAD_MAP[@]}"; do
    s3_key="${UPLOAD_MAP[${local_path}]}"
    size=$(wc -c < "${local_path}")
    echo "  ${local_path} (${size} bytes) → s3://${BUCKET_NAME}/${s3_key}"
  done
  echo "[DRY-RUN] 실제 업로드는 수행하지 않았습니다."
  exit 0
fi

# ── JSONL 형식 간단 검증 ──────────────────────────────────────────────────────
validate_jsonl() {
  local file="$1"
  local line_num=0
  local error_count=0

  while IFS= read -r line; do
    line_num=$((line_num + 1))
    if [[ -z "${line}" ]]; then
      continue
    fi
    if ! echo "${line}" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
      echo "[WARN] ${file}:${line_num} 유효하지 않은 JSON 라인"
      error_count=$((error_count + 1))
    fi
  done < "${file}"

  if [[ ${error_count} -gt 0 ]]; then
    echo "[ERROR] ${file}에서 ${error_count}개의 JSON 파싱 오류 발견"
    return 1
  fi
  return 0
}

echo "[INFO] JSONL 형식 검증 중..."
for local_path in "${!UPLOAD_MAP[@]}"; do
  echo "[INFO] 검증 중: $(basename "${local_path}")"
  validate_jsonl "${local_path}"
  echo "[OK] 형식 검증 통과: $(basename "${local_path}")"
done

# ── S3 업로드 ─────────────────────────────────────────────────────────────────
echo "[INFO] S3 업로드 시작..."
for local_path in "${!UPLOAD_MAP[@]}"; do
  s3_key="${UPLOAD_MAP[${local_path}]}"
  s3_uri="s3://${BUCKET_NAME}/${s3_key}"
  filename="$(basename "${local_path}")"
  size=$(wc -c < "${local_path}")
  line_count=$(wc -l < "${local_path}")

  echo "[INFO] 업로드 중: ${filename} (${size} bytes, ${line_count} lines) → ${s3_uri}"
  aws s3 cp "${local_path}" "${s3_uri}" \
    --region "${REGION}" \
    --profile profile2

  # 업로드 검증 (ETag 확인)
  if aws s3api head-object --bucket "${BUCKET_NAME}" --key "${s3_key}" --profile profile2 &>/dev/null; then
    echo "[OK] 업로드 확인: ${s3_uri}"
  else
    echo "[ERROR] 업로드 검증 실패: ${s3_uri}"
    exit 1
  fi
done

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  데이터 업로드 완료"
echo "============================================================"
echo "  Bucket : s3://${BUCKET_NAME}"
for local_path in "${!UPLOAD_MAP[@]}"; do
  s3_key="${UPLOAD_MAP[${local_path}]}"
  echo "  File   : s3://${BUCKET_NAME}/${s3_key}"
done
echo "============================================================"
echo ""
echo "S3 내용 확인:"
aws s3 ls "s3://${BUCKET_NAME}/data/" --profile profile2
