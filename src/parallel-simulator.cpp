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
  std::unique_ptr<QuadTreeNode> buildQuadTree(std::vector<Particle> &particles,
                                              Vec2 bmin, Vec2 bmax, int N, int* idx) {

    std::unique_ptr<QuadTreeNode> curNode(new QuadTreeNode);
    int topLeftCount = 0;
    int topRightCount = 0;
    int botLeftCount = 0;
    int botRightCount = 0;

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
        
      // Iterate over index
      for (int i = 0; i < N; i++)
      {
        int particleIdx = idx[i];
        const Particle &p = particles[particleIdx];
        bool isLeft = p.position.x < pivot.x;
        bool isUp = p.position.y < pivot.y;

        if (isLeft && isUp)
          topLeftIdx[topLeftCount++] = particleIdx;
        else if (!isLeft && isUp)
          topRightIdx[topRightCount++] = particleIdx;
        else if (isLeft && !isUp)
          botLeftIdx[botLeftCount++] = particleIdx;
        else
          botRightIdx[botRightCount++] = particleIdx;
      }

      #pragma omp parallel for schedule(static, 1)
      for (int i = 0; i < 4; i++)
      {
        if (i == 0)
          curNode->children[0] = buildQuadTree(particles, bmin, pivot, topLeftCount, topLeftIdx);
        else if (i == 1)
        {
          Vec2 topRightMin = {pivot.x, bmin.y};
          Vec2 topRightMax = {bmax.x, pivot.y};
          curNode->children[1] = buildQuadTree(particles, topRightMin, topRightMax, topRightCount, topRightIdx);
        }
        else if (i == 2)
        {
          Vec2 bottomLeftMin = {bmin.x, pivot.y};
          Vec2 bottomLeftMax = {pivot.x, bmax.y};
          curNode->children[2] = buildQuadTree(particles, bottomLeftMin, bottomLeftMax, botLeftCount, botLeftIdx);
        }
        else
          curNode->children[3] = buildQuadTree(particles, pivot, bmax, botRightCount, botRightIdx);
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

  // Do not modify this function type.
  virtual void simulateStep(AccelerationStructure *accel,
                            std::vector<Particle> &particles,
                            std::vector<Particle> &newParticles,
                            StepParameters params) override {
    // TODO: implement parallel version of quad-tree accelerated n-body
    // simulation here, using quadTree as acceleration structure
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < particles.size(); i++)
    {
      Particle curParticle = particles[i];

      // Vec2 force = Vec2(0.0f, 0.0f);
      float x_force = 0;
      float y_force = 0;
      std::vector<Particle> nearbyParticles;
      accel->getParticles(nearbyParticles, curParticle.position, params.cullRadius);

      #pragma omp parallel for reduction(+:x_force,y_force)
      for (const Particle& nearbyP : nearbyParticles)
      {
        Vec2 force = computeForce(curParticle, nearbyP, params.cullRadius);
        x_force += force.x;
        y_force += force.y;
      }

      newParticles[i] = updateParticle(curParticle, Vec2(x_force,y_force), params.deltaTime);
    }
  }
};

// Do not modify this function type.
std::unique_ptr<INBodySimulator> createParallelNBodySimulator() {
  return std::make_unique<ParallelNBodySimulator>();
}
