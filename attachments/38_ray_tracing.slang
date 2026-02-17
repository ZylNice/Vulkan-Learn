
struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
};

// UBO
struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float3 cameraPos; 
};
[[vk::binding(0, 0)]]   // 绑定点，描述符集
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(1, 0)]]
RaytracingAccelerationStructure accelerationStructure;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> indexBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<float2> uvBuffer;

struct InstanceLUT {
    uint materialID;        // 材质索引
    uint indexBufferOffset; // 索引偏移
};
[[vk::binding(4, 0)]]
StructuredBuffer<InstanceLUT> instanceLUTBuffer;

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 worldPos;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;

    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.fragNormal = input.inNormal;
    output.worldPos = mul(ubo.model, float4(input.inPosition, 1.0)).xyz;

    return output;
}

[[vk::binding(0, 1)]]
SamplerState textureSampler;

[[vk::binding(1, 1)]]
Texture2D<float4> textures[];

// 推送常量
struct PushConstant {
    uint materialIndex; // 材质索引（当前绘制）
    uint reflective;
};
[push_constant]
PushConstant pc;

static const float3 lightDir = float3(-6.0, 0.0, 6.0);

static const float EPSILON = 0.01;

// UV 插值
float2 intersection_uv(uint instanceID, uint primIndex, float2 barycentrics) {
    uint indexOffset = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].indexBufferOffset;

    uint i0 = indexBuffer[indexOffset + (primIndex * 3 + 0)];
    uint i1 = indexBuffer[indexOffset + (primIndex * 3 + 1)];
    uint i2 = indexBuffer[indexOffset + (primIndex * 3 + 2)];

    float2 uv0 = uvBuffer[i0];
    float2 uv1 = uvBuffer[i1];
    float2 uv2 = uvBuffer[i2];

    float w0 = 1.0 - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;

    return w0 * uv0 + w1 * uv1 + w2 * uv2;
}

// 应用反射
void apply_reflection(float3 P, float3 N, inout float4 baseColor) 
{
    float3 V = normalize(ubo.cameraPos - P);    // 视线方向
    float3 R = reflect(-V, N);

    RayDesc reflectionRayDesc;
    reflectionRayDesc.Origin = P;       // 起点
    reflectionRayDesc.Direction = R;    // 方向
    reflectionRayDesc.TMin = EPSILON;   // 最小距离（避免自相交）
    reflectionRayDesc.TMax = 1e4;       // 最大距离

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rq;
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

    rq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, reflectionRayDesc);
    while (rq.Proceed())
    {
        uint instanceID = rq.CandidateRayInstanceCustomIndex();
        uint primIndex = rq.CandidatePrimitiveIndex();

        float2 uv = intersection_uv(instanceID, primIndex, rq.CandidateTriangleBarycentrics());
        uint materialID = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].materialID;
        float4 intersectionColor = textures[NonUniformResourceIndex(materialID)].SampleLevel(textureSampler, uv, 0);

        if (intersectionColor.a < 0.5)
        {}
        else
        {
              rq.CommitNonOpaqueTriangleHit(); 
        }
    }

    bool hit = (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    if (hit)
    {
        uint instanceID = rq.CommittedRayInstanceCustomIndex();
        uint primIndex = rq.CommittedPrimitiveIndex();

        float2 uv = intersection_uv(instanceID, primIndex, rq.CommittedTriangleBarycentrics());
        uint materialID = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].materialID;
        float4 intersectionColor = textures[NonUniformResourceIndex(materialID)].SampleLevel(textureSampler, uv, 0);
        baseColor.rgb = lerp(baseColor.rgb, intersectionColor.rgb, 0.7);
    }
}

bool in_shadow(float3 P)
{
    RayDesc shadowRayDesc;
    shadowRayDesc.Origin = P;
    shadowRayDesc.Direction = normalize(lightDir);
    shadowRayDesc.TMin = EPSILON;
    shadowRayDesc.TMax = 1e4;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;    // 忽略程序化图元 | 找到一个交点就停止搜索
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    sq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, shadowRayDesc);
    while (sq.Proceed())
    {
        uint instanceID = sq.CandidateRayInstanceCustomIndex();
        uint primIndex = sq.CandidatePrimitiveIndex();
        float2 uv = intersection_uv(instanceID, primIndex, sq.CandidateTriangleBarycentrics());
        uint materialID = instanceLUTBuffer[NonUniformResourceIndex(instanceID)].materialID;
        float4 intersectionColor = textures[NonUniformResourceIndex(materialID)].SampleLevel(textureSampler, uv, 0);

        if (intersectionColor.a < 0.5)
        {}
        else
        {
            sq.CommitNonOpaqueTriangleHit();
        }
    }

    bool hit = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
    return hit;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target
{
    // 光栅化
    float4 baseColor = textures[pc.materialIndex].Sample(textureSampler, vertIn.fragTexCoord);

    if (baseColor.a < 0.5) discard;

    float3 P = vertIn.worldPos;
    float3 N = vertIn.fragNormal;
    if (pc.reflective > 0)
    {
        apply_reflection(P, N, baseColor);
    }

    bool inShadow = in_shadow(P);
    if (inShadow)
    {
        baseColor.rgb *= 0.2;
    }

    return baseColor;
}