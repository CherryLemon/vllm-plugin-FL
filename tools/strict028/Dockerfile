ARG BASE_IMAGE=vllm/vllm-openai:v0.28.0-cu129
FROM ${BASE_IMAGE}
ARG FL_PLUGIN_VERSION
ARG FL_GEMS_VERSION

# Build wheels with build_wheels.sh from the pinned source snapshots before
# constructing this runtime image. The vLLM wheel MUST have the +empty suffix.
COPY wheels/ /opt/fl-wheels/
COPY deployment-manifest.json /opt/fl-deployment-manifest.json
RUN test -n "${FL_PLUGIN_VERSION}" && test -n "${FL_GEMS_VERSION}" && \
    python3 -m venv --system-site-packages /opt/fl-venv && \
    uv pip install --python /opt/fl-venv/bin/python --no-index --no-deps \
      --find-links /opt/fl-wheels \
      'vllm==0.28.0+empty' "vllm-plugin-fl==${FL_PLUGIN_VERSION}" "flag-gems==${FL_GEMS_VERSION}" \
      'PyYAML==6.0.1' 'packaging==26.3' 'SQLAlchemy==2.0.54' \
      'greenlet==3.5.6' 'typing_extensions==4.16.0'
ENV PATH="/opt/fl-venv/bin:${PATH}" \
    VLLM_FL_STRICT028=1 \
    VLLM_PLUGINS=fl \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=4
RUN /opt/fl-venv/bin/python -c "import importlib.util; from vllm_fl.strict028.bootstrap import validate_host; validate_host(); assert importlib.util.find_spec('vllm._C') is None; assert importlib.util.find_spec('vllm._C_stable_libtorch') is None"
ENTRYPOINT ["/opt/fl-venv/bin/vllm", "serve"]
