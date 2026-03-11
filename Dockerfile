# CUDA 12.8 devel on Ubuntu 20.04 (includes nvcc)
FROM nvidia/cuda:12.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---------------------------
# CUDA env (make CMake/FindCUDA happy)
# ---------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ---------------------------
# Base tools + X11/GL deps
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
    build-essential cmake git pkg-config \
    wget unzip bzip2 \
    libeigen3-dev \
    # OpenGL / GLFW / offscreen (keep these; MuJoCo + RViz need them)
    libgl1-mesa-glx libgl1-mesa-dev \
    libgl1-mesa-dri \
    libglew-dev \
    libosmesa6 libosmesa6-dev \
    libglfw3 libglfw3-dev \
    # X11 runtime libs
    libx11-6 libxext6 libxrender1 \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    # System python (ROS uses it)
    python3 python3-pip python-is-python3 \
    # Debug tools
    mesa-utils x11-apps \
    # eglinfo on Ubuntu 20.04
    mesa-utils-extra \
    # basic EGL loader
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# ROS1 Noetic (keyring way, no apt-key)
# ---------------------------
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros1.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-nav-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-tf \
    ros-noetic-tf2-ros \
    ros-noetic-mavros-msgs \
    ros-noetic-mavros \
    python3-catkin-tools \
    # ROS python runtime deps for system python
    python3-yaml \
    python3-rospkg \
    python3-catkin-pkg \
    python3-empy \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# ---------------------------
# Miniforge + fixed env for MuJoCo + ROS python deps
# ---------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install Miniforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda config --system --set auto_activate_base false \
    && ${CONDA_DIR}/bin/conda config --system --set channel_priority strict \
    && ${CONDA_DIR}/bin/conda clean -afy

# Create a single runtime env (Python 3.10) and install everything needed by the bridge
RUN conda create -y -n mujoco310 python=3.10 pip \
    && conda run -n mujoco310 python -m pip install -U pip \
    && conda run -n mujoco310 python -m pip install \
    "mujoco==3.5.0" \
    pyyaml rospkg catkin_pkg empy \
    mujoco-python-viewer \
    glfw PyOpenGL \
    && conda clean -afy

# Make the env the default python in PATH (so your scripts find it without conda run)
ENV PATH=/opt/conda/envs/mujoco310/bin:/opt/conda/bin:${PATH}

# Also make interactive bash -l activate the env
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate mujoco310" >> /root/.bashrc \
    && echo "alias mujoco310='conda run -n mujoco310'" >> /root/.bashrc

# ---------------------------
# Optional: auto-fix /dev/dri render group mapping at container start
# (Avoid manual groupadd/newgrp; works even if host uses weird GID like 992)
# ---------------------------
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    '' \
    '# If /dev/dri exists, map renderD* group id to a group named "render" inside container' \
    'if [ -e /dev/dri/renderD128 ]; then' \
    '  gid="$(stat -c "%g" /dev/dri/renderD128 || true)"' \
    '  if [ -n "$gid" ]; then' \
    '    if ! getent group "$gid" >/dev/null 2>&1; then' \
    '      groupadd -g "$gid" render 2>/dev/null || true' \
    '    fi' \
    '  fi' \
    'fi' \
    '' \
    'exec "$@"' \
    > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------
# GLVND: Provide an NVIDIA EGL vendor entry (works with __EGL_VENDOR_LIBRARY_FILENAMES)
# ---------------------------
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
    '{' \
    '  "file_format_version" : "1.0.0",' \
    '  "ICD" : { "library_path" : "libEGL_nvidia.so.0" }' \
    '}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ---------------------------
# Convenience: show CUDA version quickly
# ---------------------------
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'echo "=== CUDA ==="' \
    'nvcc --version || true' \
    'echo "=== NVIDIA-SMI (host driver) ==="' \
    'nvidia-smi || true' \
    > /usr/local/bin/check_gpu.sh \
    && chmod +x /usr/local/bin/check_gpu.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash","-l"]