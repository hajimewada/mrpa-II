#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JSUT_DATA_ROOT="${JSUT_DATA_ROOT:-$HOME/projects/mrpa/data/jsut}"
JSUT_CORPUS_ROOT="${JSUT_CORPUS_ROOT:-${JSUT_DATA_ROOT}/raw/jsut_ver1.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${JSUT_DATA_ROOT}/results/mrpa-II}"
SUBSET="${SUBSET:-basic5000}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

usage() {
    cat <<EOF
Usage:
  ${SCRIPT_NAME} [--help]

Pipeline:
  WAV -> observed F0 -> interpolated F0 -> time axis -> CWT
      -> ScaleProfile -> Boundary candidates -> Nucleus candidates

Optional environment variables:
  JSUT_DATA_ROOT       Default: ${HOME}/projects/mrpa/data/jsut
  JSUT_CORPUS_ROOT     Default: \$JSUT_DATA_ROOT/raw/jsut_ver1.1
  OUTPUT_ROOT          Default: \$JSUT_DATA_ROOT/results/mrpa-II
  SUBSET               Default: basic5000
  LIMIT                Maximum utterances; 0 means all. Default: 0
  FORCE                Reprocess completed utterances when 1. Default: 0
  PYTHON               Default: \$PROJECT_ROOT/.venv/bin/python

Examples:
  LIMIT=1 scripts/process_jsut.sh
  SUBSET=basic5000 LIMIT=10 scripts/process_jsut.sh
  SUBSET=basic5000 LIMIT=1 FORCE=1 scripts/process_jsut.sh
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || die "file does not exist: $path"
}

require_directory() {
    local path="$1"
    [[ -d "$path" ]] || die "directory does not exist: $path"
}

require_executable() {
    local path="$1"
    [[ -x "$path" ]] || die "executable does not exist: $path"
}

validate_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer: $value"
}

verify_artifact() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        printf 'error: expected artifact was not created: %s\n' "$path" >&2
        return 1
    fi
}

run_stage() {
    local stage_name="$1"
    shift

    printf '[stage] %s\n' "$stage_name"
    "$@"
}

process_utterance() {
    local wav_path="$1"
    local subset="$2"
    local utterance_id
    local utterance_dir
    local log_path
    local complete_marker

    utterance_id="$(basename "$wav_path" .wav)"
    utterance_dir="${OUTPUT_ROOT}/${subset}/${utterance_id}"
    log_path="${utterance_dir}/logs/pipeline.log"
    complete_marker="${utterance_dir}/.complete"

    if [[ -f "$complete_marker" && "$FORCE" != "1" ]]; then
        log "skip completed: ${subset}/${utterance_id}"
        return 0
    fi

    mkdir -p \
        "${utterance_dir}/f0" \
        "${utterance_dir}/cwt" \
        "${utterance_dir}/scale_profile" \
        "${utterance_dir}/boundary" \
        "${utterance_dir}/nucleus" \
        "${utterance_dir}/logs"

    local observed_f0="${utterance_dir}/f0/${utterance_id}_f0.npy"
    local time_axis="${utterance_dir}/f0/${utterance_id}_f0_time.npy"
    local f0_metadata="${utterance_dir}/f0/${utterance_id}_f0_metadata.json"
    local continuous_f0="${utterance_dir}/f0/${utterance_id}_f0_interpolated.npy"
    local voicing_mask="${utterance_dir}/f0/${utterance_id}_voicing.npy"
    local interpolation_mask="${utterance_dir}/f0/${utterance_id}_interpolation_mask.npy"
    local interpolation_metadata="${utterance_dir}/f0/${utterance_id}_interpolation_metadata.json"
    local cwt_output="${utterance_dir}/cwt/${utterance_id}_cwt.npz"
    local cwt_metadata="${utterance_dir}/cwt/${utterance_id}_cwt_metadata.json"
    local profile_output="${utterance_dir}/scale_profile/${utterance_id}_scale_profile.npz"
    local profile_metadata="${utterance_dir}/scale_profile/${utterance_id}_scale_profile_metadata.json"
    local boundary_output="${utterance_dir}/boundary/${utterance_id}_boundary_candidates.npz"
    local boundary_metadata="${utterance_dir}/boundary/${utterance_id}_boundary_candidates_metadata.json"
    local nucleus_output="${utterance_dir}/nucleus/${utterance_id}_nucleus_candidates.npz"
    local nucleus_metadata="${utterance_dir}/nucleus/${utterance_id}_nucleus_candidates_metadata.json"

    rm -f "$complete_marker"
    log "start: ${subset}/${utterance_id}"

    {
        printf 'wav=%s\n' "$wav_path"
        printf 'subset=%s\n' "$subset"
        printf 'utterance_id=%s\n' "$utterance_id"
        printf 'output_directory=%s\n' "$utterance_dir"

        if ! run_stage \
            "extract_real_f0" \
            "$PYTHON" -m analysis.f0.extract_real_f0 \
            --input "$wav_path" \
            --output "$observed_f0" \
            --time-output "$time_axis" \
            --metadata-output "$f0_metadata"; then
            return 1
        fi
        verify_artifact "$observed_f0" || return 1
        verify_artifact "$time_axis" || return 1
        verify_artifact "$f0_metadata" || return 1

        if ! run_stage \
            "interpolate_f0" \
            "$PYTHON" -m analysis.f0.interpolate_f0 \
            --input "$observed_f0" \
            --time-input "$time_axis" \
            --output "$continuous_f0" \
            --voicing-output "$voicing_mask" \
            --interpolation-mask-output "$interpolation_mask" \
            --metadata-output "$interpolation_metadata" \
            --source-metadata "$f0_metadata"; then
            return 1
        fi
        verify_artifact "$continuous_f0" || return 1
        verify_artifact "$voicing_mask" || return 1
        verify_artifact "$interpolation_mask" || return 1
        verify_artifact "$interpolation_metadata" || return 1

        if ! run_stage \
            "extract_cwt" \
            "$PYTHON" -m analysis.f0.extract_cwt \
            --input "$continuous_f0" \
            --time-input "$time_axis" \
            --voicing-input "$voicing_mask" \
            --interpolation-mask-input "$interpolation_mask" \
            --interpolation-metadata "$interpolation_metadata" \
            --output "$cwt_output" \
            --metadata-output "$cwt_metadata"; then
            return 1
        fi
        verify_artifact "$cwt_output" || return 1
        verify_artifact "$cwt_metadata" || return 1

        if ! run_stage \
            "extract_scale_profile" \
            "$PYTHON" -m analysis.f0.extract_scale_profile \
            --input "$cwt_output" \
            --metadata-input "$cwt_metadata" \
            --output "$profile_output" \
            --metadata-output "$profile_metadata"; then
            return 1
        fi
        verify_artifact "$profile_output" || return 1
        verify_artifact "$profile_metadata" || return 1

        if ! run_stage \
            "extract_boundary_candidates" \
            "$PYTHON" -m analysis.boundary.extract_boundary_candidates \
            --input "$profile_output" \
            --metadata-input "$profile_metadata" \
            --output "$boundary_output" \
            --metadata-output "$boundary_metadata"; then
            return 1
        fi
        verify_artifact "$boundary_output" || return 1
        verify_artifact "$boundary_metadata" || return 1

        if ! run_stage \
            "extract_nucleus_candidates" \
            "$PYTHON" -m analysis.nucleus.extract_nucleus_candidates \
            --input "$profile_output" \
            --metadata-input "$profile_metadata" \
            --boundary-input "$boundary_output" \
            --boundary-metadata-input "$boundary_metadata" \
            --output "$nucleus_output" \
            --metadata-output "$nucleus_metadata"; then
            return 1
        fi
        verify_artifact "$nucleus_output" || return 1
        verify_artifact "$nucleus_metadata" || return 1

        printf 'pipeline_status=complete\n'
    } >"$log_path" 2>&1 || {
        printf 'failed: %s\nlog: %s\n' "$wav_path" "$log_path" >&2
        return 1
    }

    touch "$complete_marker"
    log "complete: ${subset}/${utterance_id}"
    return 0
}

main() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        usage
        return 0
    fi

    [[ $# -eq 0 ]] || die "unsupported argument: $1"

    cd "$PROJECT_ROOT"

    require_executable "$PYTHON"
    validate_nonnegative_integer "LIMIT" "$LIMIT"

    if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
        die "FORCE must be 0 or 1"
    fi

    require_directory "$PROJECT_ROOT"
    require_directory "$JSUT_DATA_ROOT"
    require_directory "$JSUT_CORPUS_ROOT"

    local wav_root="${JSUT_CORPUS_ROOT}/${SUBSET}/wav"
    require_directory "$wav_root"

    local required_script
    for required_script in \
        "${PROJECT_ROOT}/analysis/f0/extract_real_f0.py" \
        "${PROJECT_ROOT}/analysis/f0/interpolate_f0.py" \
        "${PROJECT_ROOT}/analysis/f0/extract_cwt.py" \
        "${PROJECT_ROOT}/analysis/f0/extract_scale_profile.py" \
        "${PROJECT_ROOT}/analysis/boundary/extract_boundary_candidates.py" \
        "${PROJECT_ROOT}/analysis/nucleus/extract_nucleus_candidates.py"
    do
        require_file "$required_script"
    done

    mkdir -p "$OUTPUT_ROOT"

    local processed=0
    local failed=0
    local wav_path

    while IFS= read -r -d '' wav_path; do
        if ! process_utterance "$wav_path" "$SUBSET"; then
            failed=$((failed + 1))
        fi

        processed=$((processed + 1))

        if (( LIMIT > 0 && processed >= LIMIT )); then
            break
        fi
    done < <(
        find "$wav_root" \
            -maxdepth 1 \
            -type f \
            -name '*.wav' \
            -print0 \
            | sort -z
    )

    printf 'processed=%d\n' "$processed"
    printf 'failed=%d\n' "$failed"
    printf 'output_root=%s\n' "$OUTPUT_ROOT"

    (( processed > 0 )) || die "no WAV files found under $wav_root"
    (( failed == 0 ))
}

main "$@"
