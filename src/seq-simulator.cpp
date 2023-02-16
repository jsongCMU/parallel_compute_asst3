#include "quad-tree.h"
#include "world.h"
#include <algorithm>
#include <iostream>

// TASK 1

// NOTE: You may modify any of the contents of this file, but preserve all
// function types and names. You may add new functions if you believe they will
// be helpful.

const int QuadTreeLeafSize = 8;
class SequentialNBodySimulator : public INBodySimulator
{
public:
  std::unique_ptr<QuadTreeNode> buildQuadTree(std::vector<Particle> &particles,
                                              Vec2 bmin, Vec2 bmax)
  {
    // TODO: implement a function that builds and returns a quadtree containing
    // particles.
    std::unique_ptr<QuadTreeNode> curNode(new QuadTreeNode);

    if (particles.size() <= QuadTreeLeafSize)
    { 
      curNode->isLeaf = true;
      curNode->particles = particles;
      return curNode;
    }
    else
    {
      curNode->isLeaf = false;
      Vec2 pivot;
      pivot.x = (bmax.x + bmin.x) / 2;
      pivot.y = (bmax.y + bmin.y) / 2;

      std::vector<Particle> childVectors[4];

      for(Particle &p : particles)
      {
        bool isLeft = p.position.x < pivot.x;
        bool isUp = p.position.y < pivot.y;

        if (isLeft && isUp)
          childVectors[0].push_back(p);
        else if (!isLeft && isUp)
          childVectors[1].push_back(p);
        else if (isLeft && !isUp)
          childVectors[2].push_back(p);
        else
          childVectors[3].push_back(p);
      }

      // Create the subtrees for this node
      curNode->children[0] = buildQuadTree(childVectors[0], bmin, pivot);

      Vec2 topRightMin = {pivot.x, bmin.y};
      Vec2 topRightMax = {bmax.x, pivot.y};
      curNode->children[1] = buildQuadTree(childVectors[1], topRightMin, topRightMax);

      Vec2 bottomLeftMin = {bmin.x, pivot.y};
      Vec2 bottomLeftMax = {pivot.x, bmax.y};
      curNode->children[2] = buildQuadTree(childVectors[2], bottomLeftMin, bottomLeftMax);

      curNode->children[3] = buildQuadTree(childVectors[3], pivot, bmax);

      return curNode;
    }

    return nullptr;
  }
  virtual std::unique_ptr<AccelerationStructure>
  buildAccelerationStructure(std::vector<Particle> &particles)
  {
    // build quad-tree
    auto quadTree = std::make_unique<QuadTree>();

    // find bounds
    Vec2 bmin(1e30f, 1e30f);
    Vec2 bmax(-1e30f, -1e30f);

    for (auto &p : particles)
    {
      bmin.x = fminf(bmin.x, p.position.x);
      bmin.y = fminf(bmin.y, p.position.y);
      bmax.x = fmaxf(bmax.x, p.position.x);
      bmax.y = fmaxf(bmax.y, p.position.y);
    }

    quadTree->bmin = bmin;
    quadTree->bmax = bmax;

    // build nodes
    quadTree->root = buildQuadTree(particles, bmin, bmax);
    if (!quadTree->checkTree())
    {
      std::cout << "Your Tree has Error!" << std::endl;
    }

    return quadTree;
  }
  virtual void simulateStep(AccelerationStructure *accel,
                            std::vector<Particle> &particles,
                            std::vector<Particle> &newParticles,
                            StepParameters params) override
  {
    // TODO: implement sequential version of quad-tree accelerated n-body
    // simulation here, using quadTree as acceleration structure
    for (int i = 0; i < particles.size(); i++)
    {
      Particle curParticle = particles[i];

      Vec2 force = Vec2(0.0f, 0.0f);
      std::vector<Particle> nearbyParticles;
      accel->getParticles(nearbyParticles, curParticle.position, params.cullRadius);

      for (const Particle& nearbyP : nearbyParticles)
        force += computeForce(curParticle, nearbyP, params.cullRadius);

      newParticles[i] = updateParticle(curParticle, force, params.deltaTime);
    }
  }
};

std::unique_ptr<INBodySimulator> createSequentialNBodySimulator()
{
  return std::make_unique<SequentialNBodySimulator>();
}
