Package: ktx:x64-windows@4.3.2#2

**Host Environment**

- Host: x64-windows
- Compiler: MSVC 19.44.35215.0
-    vcpkg-tool version: 2025-07-21-d4b65a2b83ae6c3526acd1c6f3b51aff2a884533
    vcpkg-scripts version: unknown

**To Reproduce**

`vcpkg install `

**Failure logs**

```
Downloading https://github.com/KhronosGroup/KTX-Software/archive/v4.3.2.tar.gz -> KhronosGroup-KTX-Software-v4.3.2.tar.gz
KhronosGroup-KTX-Software-v4.3.2.tar.gz.31840.part: error: download from https://github.com/KhronosGroup/KTX-Software/archive/v4.3.2.tar.gz had an unexpected hash
note: Expected: 0077315fe2b4e676e97e3a158c2c6e1f6ba426e14ad23342592cd69be28cfce64c40614e0a84d58a9634877ab334e713b94d4c962132c98bfea308e91bc8a98a
note: Actual  : 0979cc4f24566e6e0fe233911275816a7b19a03b72bc1e06044f65d53e525218238a040d8551279d359f7151af1d383f81198b644c0667c5bc0aa2a5f96a8b8d
CMake Error at scripts/cmake/vcpkg_download_distfile.cmake:136 (message):
  Download failed, halting portfile.
Call Stack (most recent call first):
  scripts/cmake/vcpkg_from_github.cmake:120 (vcpkg_download_distfile)
  ports/ktx/portfile.cmake:1 (vcpkg_from_github)
  scripts/ports.cmake:206 (include)



```

**Additional context**

<details><summary>vcpkg.json</summary>

```
{
  "name": "vulkan-tutorial",
  "version": "1.0.0",
  "dependencies": [
    "glfw3",
    "glm",
    "tinyobjloader",
    "stb",
    "tinygltf",
    "nlohmann-json",
    "ktx"
  ]
}

```
</details>
