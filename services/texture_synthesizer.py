import numpy as np, heapq, os, math, cv2
from PIL import Image
from skimage import io, util
from model import QuiltMode


class TextureSynthesizer(object):
    def __init__(self, num_base_texture_blocks, overlap_ratio=6):
        self.num_base_texture_blocks = num_base_texture_blocks
        self.overlap_ratio = overlap_ratio

    def randomPatch(self, texture, block_size):
        h, w, _ = texture.shape
        i = np.random.randint(h - block_size)
        j = np.random.randint(w - block_size)

        return texture[i:i + block_size, j:j + block_size]

    def L2OverlapDiff(self, patch, block_size, overlap, res, y, x):
        error = 0
        if x > 0:
            left = patch[:, :overlap] - res[y:y + block_size, x:x + overlap]
            error += np.sum(left**2)

        if y > 0:
            up = patch[:overlap, :] - res[y:y + overlap, x:x + block_size]
            error += np.sum(up**2)

        if x > 0 and y > 0:
            corner = patch[:overlap, :overlap] - res[y:y + overlap,
                                                     x:x + overlap]
            error -= np.sum(corner**2)

        return error

    def randomBestPatch(self, texture, block_size, overlap, res, y, x):
        h, w, _ = texture.shape
        errors = np.zeros((h - block_size, w - block_size))

        for i in range(h - block_size):
            for j in range(w - block_size):
                patch = texture[i:i + block_size, j:j + block_size]
                e = self.L2OverlapDiff(patch, block_size, overlap, res, y, x)
                errors[i, j] = e

        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        return texture[i:i + block_size, j:j + block_size]

    def minCutPath(self, errors):
        # dijkstra's algorithm vertical
        pq = [(error, [i]) for i, error in enumerate(errors[0])]
        heapq.heapify(pq)

        h, w = errors.shape
        seen = set()

        while pq:
            error, path = heapq.heappop(pq)
            curDepth = len(path)
            curIndex = path[-1]

            if curDepth == h:
                return path

            for delta in -1, 0, 1:
                nextIndex = curIndex + delta

                if 0 <= nextIndex < w:
                    if (curDepth, nextIndex) not in seen:
                        cumError = error + errors[curDepth, nextIndex]
                        heapq.heappush(pq, (cumError, path + [nextIndex]))
                        seen.add((curDepth, nextIndex))

    def minCutPatch(self, patch, block_size, overlap, res, y, x):
        patch = patch.copy()
        dy, dx, _ = patch.shape
        minCut = np.zeros_like(patch, dtype=bool)

        if x > 0:
            left = patch[:, :overlap] - res[y:y + dy, x:x + overlap]
            leftL2 = np.sum(left**2, axis=2)
            for i, j in enumerate(self.minCutPath(leftL2)):
                minCut[i, :j] = True

        if y > 0:
            up = patch[:overlap, :] - res[y:y + overlap, x:x + dx]
            upL2 = np.sum(up**2, axis=2)
            for j, i in enumerate(self.minCutPath(upL2.T)):
                minCut[:i, j] = True

        np.copyto(patch, res[y:y + dy, x:x + dx], where=minCut)

        return patch

    def synthesizeTexture(self, base_texture, desiredShape):
        assert base_texture.shape[1] == base_texture.shape[
            0]  # for now  texture has to be a square
        block_size = math.floor(base_texture.shape[0] /
                                self.num_base_texture_blocks
                                )  # number of times we cut base_texture
        overlap = block_size // self.overlap_ratio
        h_block, w_block = self.calculateNumBlock(block_size, overlap,
                                                  desiredShape[0],
                                                  desiredShape[1])
        raw_quilted_texture = self.quilt(base_texture, block_size,
                                         (h_block, w_block), QuiltMode.BEST)
        return cv2.resize(
            raw_quilted_texture,
            (desiredShape[1],
             desiredShape[0]))  # TODO: allow for crop strategy as well

    def calculateSynthesizedTextureSize(self, block_size, overlap, h_block,
                                        w_block):
        h = (h_block * block_size) - (h_block - 1) * overlap
        w = (w_block * block_size) - (w_block - 1) * overlap
        return (h, w)

    def calculateNumBlock(self, block_size, overlap, desired_h, desired_w):
        # solve equation in calculatSynthesizedTextureSize() for h_block and w_block
        h_blocks = math.ceil((desired_h - overlap) / (block_size - overlap))
        w_blocks = math.ceil((desired_w - overlap) / (block_size - overlap))
        return (h_blocks, w_blocks)

    def quilt(self,
              texture,
              block_size,
              num_block,
              mode: QuiltMode,
              sequence=False):
        texture = util.img_as_float(texture)

        overlap = block_size // self.overlap_ratio
        num_blockHigh, num_blockWide = num_block
        h, w = self.calculateSynthesizedTextureSize(block_size, overlap,
                                                    num_blockHigh,
                                                    num_blockWide)

        res = np.zeros((h, w, texture.shape[2]))
        print(num_blockHigh, num_blockWide)
        for i in range(num_blockHigh):
            for j in range(num_blockWide):
                y = i * (block_size - overlap)
                x = j * (block_size - overlap)

                if i == 0 and j == 0 or mode == QuiltMode.RANDOM:
                    patch = self.randomPatch(texture, block_size)
                elif mode == QuiltMode.BEST:
                    patch = self.randomBestPatch(texture, block_size, overlap,
                                                 res, y, x)
                elif mode == QuiltMode.CUT:
                    patch = self.randomBestPatch(texture, block_size, overlap,
                                                 res, y, x)
                    patch = self.minCutPatch(patch, block_size, overlap, res,
                                             y, x)

                res[y:y + block_size, x:x + block_size] = patch

        return (res * 255).astype(np.uint8)
