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
  void insert(QuadTreeNode* node, Particle &p, Vec2 bmin, Vec2 bmax)
  {
    Vec2 pivot;
    Vec2 topLeft = bmin;
    Vec2 botRight = bmax;

    Vec2 prevTopLeft = bmin;
    Vec2 prevBotRight = bmax;
    

    QuadTreeNode* prev_node = nullptr;

    while (node != nullptr)
    {
      prev_node = node;

      pivot.x = (botRight.x + topLeft.x) / 2;
      pivot.y = (botRight.y + topLeft.y) / 2;

      bool isLeft = p.position.x < pivot.x;
      bool isUp = p.position.y < pivot.y;

      if (isLeft && isUp)
      {
        node = node->children[0].get();

        prevTopLeft = topLeft;
        prevBotRight = botRight;

        botRight = pivot;
      }
      else if (!isLeft && isUp)
      {
        node = node->children[1].get();

        prevTopLeft = topLeft;
        prevBotRight = botRight;

        topLeft = {pivot.x, topLeft.y};
        botRight = {botRight.x, pivot.y};
      }
      else if (isLeft && !isUp)
      {
        node = node->children[2].get();
        
        prevTopLeft = topLeft;
        prevBotRight = botRight;

        topLeft = {topLeft.x, pivot.y};
        botRight = {pivot.x, botRight.y};
      }
      else
      {
        node = node->children[3].get();

        prevTopLeft = topLeft;
        prevBotRight = botRight;

        topLeft = pivot;
      }
    }
    node = prev_node;

    if (node->particles.size() > QuadTreeLeafSize)
    {
      node->isLeaf = false;
      
      // Create new children
      for (int i = 0; i < 4; i++)
      {
        node->children[i] = std::unique_ptr<QuadTreeNode>(new QuadTreeNode);
        node->children[i]->isLeaf = true;
      }

      for (int i = 0; i < QuadTreeLeafSize; i++)
      {
        insert(node, node->particles[i], prevTopLeft, prevBotRight);
      }

      insert(node, p, prevTopLeft, prevBotRight);

      node->particles.clear();
    }
    else
      node->particles.push_back(p);
  }

  // TODO: implement a function that builds and returns a quadtree containing
  // particles. You do not have to preserve this function type.
  std::unique_ptr<QuadTreeNode> buildQuadTree(std::vector<Particle> &particles,
                                              Vec2 bmin, Vec2 bmax, int level = 0) {

    std::unique_ptr<QuadTreeNode> curNode(new QuadTreeNode);
    
    for (int i = 0; i < particles.size(); i++)
      insert(curNode.get(), particles[i], bmin, bmax);

    return curNode;
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

    // build nodes
    quadTree->root = buildQuadTree(particles, bmin, bmax);
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
