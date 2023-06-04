.. code::

    wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
    tar -xf blender-3.2.2-linux-x64.tar.xz
    rm blender-3.2.2-linux-x64.tar.xz
    cd blender-3.2.2-linux-x64/3.2/python/bin
    ./python3.10 -m ensurepip
    ./python3.10 -m pip install numpy==1.23.1 scipy
