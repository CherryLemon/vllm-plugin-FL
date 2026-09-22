#!/usr/bin/env bash
# Run from a prepared delivery. Pin the actual local base image, not a mutable tag.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
read -r base_tag base_id plugin_version gems_version < <(
  python3 - "$task_root/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
print(m['host']['base_image'],m['host']['base_image_id'],
      m['package_versions']['plugin'],m['package_versions']['FlagGems'])
PY
)
actual_base="$(docker image inspect --format '{{.Id}}' "$base_tag")"
test "$actual_base" = "$base_id"
runtime_tag="${FL_RUNTIME_TAG:-local/dsv41-fl:strict028}"
docker build --build-arg "BASE_IMAGE=$base_id" \
  --build-arg "FL_PLUGIN_VERSION=$plugin_version" \
  --build-arg "FL_GEMS_VERSION=$gems_version" \
  --iidfile "$task_root/evidence/runtime-image.id" \
  -t "$runtime_tag" "$task_root/image" \
  2>&1 | tee "$task_root/evidence/image-build.log"
python3 - "$task_root" "$runtime_tag" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); m=json.loads((root/'manifest.json').read_text())
receipt={'status':'built','tag':sys.argv[2],
         'image_id':(root/'evidence/runtime-image.id').read_text().strip(),
         'base_image_id':m['host']['base_image_id'],
         'package_versions':m['package_versions'],
         'model_inference_in_this_image':False}
(root/'evidence/image-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
