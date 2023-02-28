#include "quad-tree.h"
#include "world.h"
#include <algorithm>
#include <iostream>
#include <omp.h>

// TASK 2

// NOTE: You may modify this class definition as you see fit, as long as the
// class name, and type of simulateStep and buildAccelerationStructure remain
// the same. You may modify any code outside this class unless otherwise
// specified.

const int QuadTreeLeafSize = 32;
class ParallelNBodySimulator : public INBodySimulator {
public:
  // TODO: implement a function that builds and returns a quadtree containing
  // particles. You do not have to preserve this function type.
  std::unique_ptr<QuadTreeNode> buildQuadTree(const std::vector<Particle> &particles,
                                              Vec2 bmin, Vec2 bmax, int N, int* idx, int level = 0) {

    std::unique_ptr<QuadTreeNode> curNode(new QuadTreeNode);

    if (N <= QuadTreeLeafSize)
    {
      curNode->isLeaf = true;
      curNode->particles.reserve(N);
      
      for (int i = 0; i < N; i++)
        curNode->particles.push_back(particles[idx[i]]);

      return curNode;
    }
    else
    {
      curNode->isLeaf = false;
      Vec2 pivot;
      pivot.x = (bmax.x + bmin.x) / 2;
      pivot.y = (bmax.y + bmin.y) / 2;

      int topLeftIdx[N];
      int topRightIdx[N];
      int botLeftIdx[N];
      int botRightIdx[N];
      int topLeftCount = 0;
      int topRightCount = 0;
      int botLeftCount = 0;
      int botRightCount = 0;

      int* offsets[4];
      #pragma omp parallel
      {
        // Compute indexes for each thread
        int localTopLeftIdx[N];
        int localTopRightIdx[N];
        int localBotLeftIdx[N];
        int localBotRightIdx[N];
        int localTopLeftCount = 0;
        int localTopRightCount = 0;
        int localBotLeftCount = 0;
        int localBotRightCount = 0;
        #pragma omp for nowait
        for (int i = 0; i < N; i++)
        {
            int particleIdx = idx[i];
            const Particle &p = particles[particleIdx];
            bool isLeft = p.position.x < pivot.x;
            bool isUp = p.position.y < pivot.y;

            if (isLeft && isUp)
                localTopLeftIdx[localTopLeftCount++] = particleIdx;
            else if (!isLeft && isUp)
                localTopRightIdx[localTopRightCount++] = particleIdx;
            else if (isLeft && !isUp)
                localBotLeftIdx[localBotLeftCount++] = particleIdx;
            else
                localBotRightIdx[localBotRightCount++] = particleIdx;
        }
        // Compute offsets via scans
        #pragma omp single
        {
            offsets[0] = new int[omp_get_num_threads()+1];
            offsets[1] = new int[omp_get_num_threads()+1];
            offsets[2] = new int[omp_get_num_threads()+1];
            offsets[3] = new int[omp_get_num_threads()+1];
            offsets[0][0] = 0;
            offsets[1][0] = 0;
            offsets[2][0] = 0;
            offsets[3][0] = 0;
        }
        offsets[0][omp_get_thread_num()+1] = localTopLeftCount;
        offsets[1][omp_get_thread_num()+1] = localTopRightCount;
        offsets[2][omp_get_thread_num()+1] = localBotLeftCount;
        offsets[3][omp_get_thread_num()+1] = localBotRightCount;
        #pragma omp barrier
        #pragma omp for
        for(int i = 0; i < 4; i++)
        {
            for(int j = 1; j<omp_get_num_threads()+1; j++)
            {
                offsets[i][j] += offsets[i][j-1];
            }
        }
        // Combine indexes across threads
        #pragma omp single
        {
            topLeftCount = offsets[0][omp_get_num_threads()];
            topRightCount = offsets[1][omp_get_num_threads()];
            botLeftCount = offsets[2][omp_get_num_threads()];
            botRightCount = offsets[3][omp_get_num_threads()];
        }
        std::move(localTopLeftIdx, localTopLeftIdx+localTopLeftCount, topLeftIdx+offsets[0][omp_get_thread_num()]);
        std::move(localTopRightIdx, localTopRightIdx+localTopRightCount, topRightIdx+offsets[1][omp_get_thread_num()]);
        std::move(localBotLeftIdx, localBotLeftIdx+localBotLeftCount, botLeftIdx+offsets[2][omp_get_thread_num()]);
        std::move(localBotRightIdx, localBotRightIdx+localBotRightCount, botRightIdx+offsets[3][omp_get_thread_num()]);
      }
      delete[] offsets[0];
      delete[] offsets[1];
      delete[] offsets[2];
      delete[] offsets[3];

      // Stop at level 5 or if the number of threads is too low
      if (level <= 5 || omp_get_max_threads() < 4)
      {
        #pragma omp parallel
        {
          #pragma omp single
          {
            #pragma omp task untied default(none) shared(curNode, particles, bmin, pivot, topLeftCount,topLeftIdx, level)
            curNode->children[0] = buildQuadTree(particles, bmin, pivot, topLeftCount, topLeftIdx, level + 1);

            Vec2 topRightMin = {pivot.x, bmin.y};
            Vec2 topRightMax = {bmax.x, pivot.y};
            #pragma omp task untied default(none) shared(curNode, particles, topRightMin, topRightMax, topRightCount, topRightIdx, level)
            curNode->children[1] = buildQuadTree(particles, topRightMin, topRightMax, topRightCount, topRightIdx, level + 1);
            
            Vec2 bottomLeftMin = {bmin.x, pivot.y};
            Vec2 bottomLeftMax = {pivot.x, bmax.y};
            #pragma omp task untied default(none) shared(curNode, particles, bottomLeftMin, bottomLeftMax, botLeftCount, botLeftIdx, level)
            curNode->children[2] = buildQuadTree(particles, bottomLeftMin, bottomLeftMax, botLeftCount, botLeftIdx, level + 1);

            #pragma omp task untied default(none) shared(curNode, particles, pivot, bmax, botRightCount, botRightIdx, level)
            curNode->children[3] = buildQuadTree(particles, pivot, bmax, botRightCount, botRightIdx, level + 1);
          }
        }
      }
      else 
      {
        curNode->children[0] = buildQuadTree(particles, bmin, pivot, topLeftCount, topLeftIdx, level + 1);

        Vec2 topRightMin = {pivot.x, bmin.y};
        Vec2 topRightMax = {bmax.x, pivot.y};
        curNode->children[1] = buildQuadTree(particles, topRightMin, topRightMax, topRightCount, topRightIdx, level + 1);
        
        Vec2 bottomLeftMin = {bmin.x, pivot.y};
        Vec2 bottomLeftMax = {pivot.x, bmax.y};
        curNode->children[2] = buildQuadTree(particles, bottomLeftMin, bottomLeftMax, botLeftCount, botLeftIdx, level + 1);

        curNode->children[3] = buildQuadTree(particles, pivot, bmax, botRightCount, botRightIdx, level + 1);
      }

      return curNode;
    }

    return nullptr;
  }

  // Do not modify this function type.
  virtual std::unique_ptr<AccelerationStructure>
  buildAccelerationStructure(std::vector<Particle> &particles) {
    // build quad-tree
    auto quadTree = std::make_unique<QuadTree>();

    // find bounds
    Vec2 bmin(1e30f, 1e30f);
    Vec2 bmax(-1e30f, -1e30f);

    for (auto &p : particles) {
      bmin.x = fminf(bmin.x, p.position.x);
      bmin.y = fminf(bmin.y, p.position.y);
      bmax.x = fmaxf(bmax.x, p.position.x);
      bmax.y = fmaxf(bmax.y, p.position.y);
    }

    quadTree->bmin = bmin;
    quadTree->bmax = bmax;

    int idx[particles.size()];

    for(int i = 0; i < particles.size(); i++)
      idx[i] = i;
    
    // build nodes
    quadTree->root = buildQuadTree(particles, bmin, bmax, particles.size(), idx);
    if (!quadTree->checkTree()) {
      std::cout << "Your Tree has Error!" << std::endl;
    }

    return quadTree;
  }

  #pragma omp declare reduction(Vec2Plus: Vec2: omp_out += omp_in)

  // Do not modify this function type.
  virtual void simulateStep(AccelerationStructure *accel,
                            std::vector<Particle> &particles,
                            std::vector<Particle> &newParticles,
                            StepParameters params) override {
    // TODO: implement parallel version of quad-tree accelerated n-body
    // simulation here, using quadTree as acceleration structure
    #pragma omp parallel for schedule(dynamic, (particles.size() + 63)/64)
    for (int i = 0; i < particles.size(); i++)
    {
      Particle curParticle = particles[i];

      Vec2 force = Vec2(0.0f, 0.0f);
      std::vector<Particle> nearbyParticles;
      accel->getParticles(nearbyParticles, curParticle.position, params.cullRadius);

      #pragma omp parallel for reduction(Vec2Plus:force)
      for (const Particle& nearbyP : nearbyParticles)
      {
        force += computeForce(curParticle, nearbyP, params.cullRadius);
      }

      newParticles[i] = updateParticle(curParticle, force, params.deltaTime);
    }
  }
};

// Do not modify this function type.
std::unique_ptr<INBodySimulator> createParallelNBodySimulator() {
  return std::make_unique<ParallelNBodySimulator>();
}
