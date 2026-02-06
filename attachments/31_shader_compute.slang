struct VSInput {
    float2 inPosition;
    float4 inColor;
};

struct VSOutput {
    float4 pos : SV_Position; 
    float pointSize : SV_PointSize;
    float3 fragColor : COLOR0;
};

struct PSInput {
    float4 pos : SV_POSITION;
    float3 fragColor : COLOR0;
    float2 pointCoord : SV_PointCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pointSize = 14.0;
    output.pos = float4(input.inPosition, 1.0, 1.0);
    output.fragColor = input.inColor.rgb;
    return output;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_TARGET {
    float2 coord = input.pointCoord - float2(0.5);
    return float4(input.fragColor, 0.5 - length(coord));
}

struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
};

struct UniformBuffer {
    float deltaTime;
};
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants{
    uint startIndex;
    uint count;
};
[[vk::push_constant]] PushConstants pushConstants;

struct ParticleSSBO {
    Particle particles;
};

StructuredBuffer<ParticleSSBO> particlesIn;

RWStructuredBuffer<ParticleSSBO> particlesOut;

[shader("compute")]
[numthreads(256, 1, 1)]
void compMain(uint3 threadId: SV_DispatchThreadID)
{
    uint localIndex = threadId.x;
    if (localIndex < pushConstants.count)
    {    
        uint globalIndex = pushConstants.startIndex + localIndex;
        particlesOut[globalIndex].particles.position = particlesIn[globalIndex].particles.position + particlesIn[globalIndex].particles.velocity.xy * ubo.deltaTime;
        particlesOut[globalIndex].particles.velocity = particlesIn[globalIndex].particles.velocity;

        if ((particlesOut[globalIndex].particles.position.x <= -1.0) || (particlesOut[globalIndex].particles.position.x >= 1.0)) {
            particlesOut[globalIndex].particles.velocity.x = -particlesOut[globalIndex].particles.velocity.x;
        }
        if ((particlesOut[globalIndex].particles.position.y <= -1.0) || (particlesOut[globalIndex].particles.position.y >= 1.0)) {
            particlesOut[globalIndex].particles.velocity.y = -particlesOut[globalIndex].particles.velocity.y;
        }
    }
}