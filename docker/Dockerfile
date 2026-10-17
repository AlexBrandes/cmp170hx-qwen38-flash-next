# Engine image: the official vLLM v0.30.0 image plus the changed Python files
# of the fork (AlexBrandes/vllm, branch cmp170hx-qwen38-flash-next). The fork
# changes no compiled code, so nothing is rebuilt. No weights and no launch
# settings are baked in; serve.sh supplies them.
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
ARG FORK_COMMIT
COPY overlay/ /usr/local/lib/python3.12/dist-packages/
COPY overlay.files /opt/cmp170hx/overlay.files
RUN cd /usr/local/lib/python3.12/dist-packages \
    && python3 -m compileall -q $(cat /opt/cmp170hx/overlay.files)
LABEL org.opencontainers.image.source="https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next" \
      org.opencontainers.image.description="vLLM v0.30.0 + AlexBrandes/vllm cmp170hx-qwen38-flash-next for Qwen3.8-Flash-Next on 2x CMP 170HX" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${FORK_COMMIT}"
ENTRYPOINT []
