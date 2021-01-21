#include "DevBetterDipole.h"

namespace cukd {
namespace device {

template<typename Point>
struct KdTreeTraverseData {
    IndexedPtr<PreorderTreeOpaqueElement, PreorderTreeOpaqueIndex> preorder;
    typename Point::RawArray elements;
    typename Point::RawArray avgElements;

    NodeDepth maxDepth;

    __device__
    KdTreeTraverseData(IndexedPtr<PreorderTreeOpaqueElement, PreorderTreeOpaqueIndex> preorder,
                       const typename Point::RawArray& elements,
                       const typename Point::RawArray& avgElements,
                       NodeDepth maxDepth)
        : preorder(preorder), elements(elements), avgElements(avgElements), maxDepth(maxDepth) {}
};

using KdTreeTraverseDataIP = KdTreeTraverseData<IlluminationPoint>;

struct KdTreeTraverseDataSP : KdTreeTraverseData<ShadingPoint> {

    IndexedPtr<ElementCountAndIsSmall, NodeIndex> sizeAndIsSmall;
    IndexedPtr<PixelIndex, NodeIndex> pixelOffsets;
    IndexedPtr<PackedPixel, PixelIndex> pixels;
    
    __device__
    KdTreeTraverseDataSP(IndexedPtr<PreorderTreeOpaqueElement, PreorderTreeOpaqueIndex> preorder,
                         const ShadingPoint::RawArray& elements,
                         const ShadingPoint::RawArray& avgElements,
                         IndexedPtr<ElementCountAndIsSmall, NodeIndex> sizeAndIsSmall,
                         IndexedPtr<PixelIndex, NodeIndex> pixelOffsets,
                         IndexedPtr<PackedPixel, PixelIndex> pixels,
                         NodeDepth maxDepth)
        : KdTreeTraverseData(preorder, elements, avgElements, maxDepth),
          sizeAndIsSmall(sizeAndIsSmall), pixelOffsets(pixelOffsets), pixels(pixels) {}
};

__device__
inline void updateNodeAABB(const PreorderInnerNode* node, UAABB& box, bool goingRight) {
    if(node->isLeaf())
        return;

    if(goingRight)
        sliceT(node->splitAxis(), box.minimum) = node->splitPosition();
    else
        sliceT(node->splitAxis(), box.maximum) = node->splitPosition();
}

__device__
inline bool overlaps(const UAABB& a, const UAABB& b) {
    return interval_overlap(a.minimum.vec.x, a.maximum.vec.x, b.minimum.vec.x, b.maximum.vec.x)
        && interval_overlap(a.minimum.vec.y, a.maximum.vec.y, b.minimum.vec.y, b.maximum.vec.y)
        && interval_overlap(a.minimum.vec.z, a.maximum.vec.z, b.minimum.vec.z, b.maximum.vec.z);
}

__device__
inline void shadeNodes(const DevSubsurface* ss, float4* output, int outputPitch,
                       const PreorderNode* shadeNode, const PreorderNode* illumNode,
                       const KdTreeTraverseDataSP* shade,
                       const KdTreeTraverseDataIP* illum) {
    const UFloat4& shadePosition = shade->avgElements.positions[shadeNode->elementBackReference()];
    const float3& shadeNormal = shade->avgElements.normals[shadeNode->elementBackReference()];
    const float3& shadeDirection = shade->avgElements.directions[shadeNode->elementBackReference()];

    const UFloat4& illumPosition = illum->avgElements.positions[illumNode->elementBackReference()];
    const float3& illumIrradiance = illum->avgElements.irradiances[illumNode->elementBackReference()];
    const float& illumArea = illum->avgElements.areas[illumNode->elementBackReference()];
    float3 contrib = ss->query(shadePosition, shadeNormal, shadeDirection,
                     illumPosition, illumIrradiance, illumArea);

    PixelIndex pixelsStart = shade->pixelOffsets[shadeNode->backReference()];
    PixelCount pixelsCount = shadeNode->pixelCount(shade->sizeAndIsSmall);

    for(PixelIndex i = pixelsStart; i <= pixelsStart + pixelsCount; ++i) {
        int2 px = int2(shade->pixels[i]);
        float4* outputElement = (float4*)((char*)output + px.y * outputPitch) + px.x;
        *outputElement += make_float4(contrib, 0.f);
    }
}

__global__
void traverseSubtreeDual(const DevSubsurface* ss, float4* output, int outputPitch,
                         const KdTreeTraverseDataSP* shade, const KdTreeTraverseDataIP* illum,
                         float threshold, NodeDepth depth,
                         PreorderTreeOpaqueIndex baseShadeNodeIdx, PreorderTreeOpaqueIndex baseIllumNodeIdx,
                         UAABB shadeBox, UAABB illumBox) {
    int shadeMask = threadIdx.x;
    int illumMask = threadIdx.y;
    
    PreorderTreeOpaqueIndex shadeNodeIdx = baseShadeNodeIdx;
    PreorderTreeOpaqueIndex illumNodeIdx = baseIllumNodeIdx;

    for(int currentMask = 1; currentMask <= kDualTreeChunkDepth; currentMask <<= 1, ++depth) {
    
        const auto shadeNode = nodeT(shade->preorder[shadeNodeIdx]);
        const auto illumNode = nodeT(illum->preorder[illumNodeIdx]);

        bool shadeGoingRight = (shadeMask & currentMask) != 0;
        bool illumGoingRight = (illumMask & currentMask) != 0;
        updateNodeAABB(shadeNode->asInner(), shadeBox, shadeGoingRight);
        updateNodeAABB(illumNode->asInner(), illumBox, illumGoingRight);

        if(shadeNode->isLeaf() && illumNode->isLeaf()) {
            shadeNodes(ss, output, outputPitch, shadeNode, illumNode, shade, illum);
            return;
        }

        float irrArea = illum->avgElements.areas[illumNode->elementBackReference()];
        float shadeArea = threshold * squared_distance_ufloat4(shade->avgElements.positions[shadeNode->elementBackReference()], illum->avgElements.positions[illumNode->elementBackReference()]);

        if(irrArea < shadeArea && overlaps(shadeBox, illumBox)) {
            shadeNodes(ss, output, outputPitch, shadeNode, illumNode, shade, illum);
            return;
        }

        if(!illumNode->isLeaf())
            illumNodeIdx = illumGoingRight ? illumNode->asInner()->rightIndex() : illumNodeIdx + 3_ptoi;
        if(!shadeNode->isLeaf())
            shadeNodeIdx = shadeGoingRight ? shadeNode->asInner()->rightIndex() : shadeNodeIdx + 3_ptoi;
    }

    // We know for a fact at least one node is not a leaf if we reached this point
    dim3 grid(1,1,1);
    dim3 blocks(min(int(shade->maxDepth - depth), 1 << kDualTreeChunkDepth),min(int(illum->maxDepth - depth), 1 << kDualTreeChunkDepth),1);
    traverseSubtreeDual CU_OPT(grid,blocks)(ss, output, outputPitch, shade, illum, threshold, depth, shadeNodeIdx, illumNodeIdx, shadeBox, illumBox);
}

}


KdTreeTraverser::KdTreeTraverser(std::shared_ptr<Subsurface> subsurface, std::shared_ptr<KdTree<ShadingPoint>> shadingTree, std::shared_ptr<KdTree<IlluminationPoint>> illuminationTree)
    : m_subsurface(subsurface), m_shadeTree(shadingTree), m_illumTree(illuminationTree) { }

void KdTreeTraverser::traverse(float4* outputMemory, const uint3& dims) {
    // Spawn NxN blocks, with N a power of two
    // Each bit of N marks whether the thread should walk down the left or right child for their respective tree
    // When the thread reaches a halting threshold, it performs the relevant computation
    // When the thread reaches its maximum depth (i.e. log_2 N), it spawns another block for its node
    // Should a leaf be reached, only the thread with zeroes for everything after the leaf's bit mask (i.e. the one
    //  which would continue down left/left children only) continues, the others return
    // Start with 1 block and go from there
    // 16x16 blocks are probably ideal

    DevObject<device::KdTreeTraverseDataSP> shade;
    shade.makeDevice(m_shadeTree->m_preorderTree.pointer(), m_shadeTree->m_KDTreeNWA.m_points.rawArray(), m_shadeTree->m_preorderInnerTreeElements.rawArray(), m_shadeTree->m_preorderBandwidthSizeAndIsSmall.pointer(), m_shadeTree->m_preorderBandwidthOffsets.pointer(), m_shadeTree->m_preorderPixels.pointer(), m_shadeTree->maxDepth());

    DevObject<device::KdTreeTraverseDataIP> illum;
    illum.makeDevice(m_illumTree->m_preorderTree.pointer(), m_illumTree->m_KDTreeNWA.m_points.rawArray(), m_illumTree->m_preorderInnerTreeElements.rawArray(), m_illumTree->maxDepth());

    UAABB shadeBox = m_shadeTree->m_rootAABB, illumBox = m_illumTree->m_rootAABB;
    
    auto ss = DevBetterDipole::create(static_cast<BetterDipole*>(m_subsurface.get()));

    cudaMemset2D(outputMemory, dims.z, 0, dims.x, dims.y);

    device::traverseSubtreeDual CU_OPT(1,1)(ss.devPointer(), outputMemory, dims.z,
                                                 shade.devPointer(), illum.devPointer(), m_threshold, 0, 0, 0,
                                          shadeBox, illumBox);
    reportCudaErrorsForFunction("traverseSubtreeDual failed");
}



}