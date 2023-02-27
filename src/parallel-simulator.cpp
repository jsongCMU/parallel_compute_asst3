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
                                              Vec2 bmin, Vec2 bmax) {

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
      
      std::vector<Particle> totalChildVectors[4];
    //   totalChildVectors[0].reserve(particles.size());
    //   totalChildVectors[1].reserve(particles.size());
    //   totalChildVectors[2].reserve(particles.size());
    //   totalChildVectors[3].reserve(particles.size());
      int* offsets[4];
      #pragma omp parallel
      {
        std::vector<Particle> childVectors[4];
        #pragma omp for nowait schedule(dynamic, 32)
        for(int idx = 0; idx < particles.size(); idx++)
        {
            const Particle &p = particles[idx];
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
        // Scan to determine offset for placing in global vector
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
        for(int i=0; i<4; i++)
        {
            offsets[i][omp_get_thread_num()+1] = childVectors[i].size();
        }
        #pragma omp barrier
        #pragma omp for
        for(int i = 0; i < 4; i++) // TODO: parallelize outer loop since independent
        {
            for(int j = 1; j<omp_get_num_threads()+1; j++)
            {
                offsets[i][j] += offsets[i][j-1];
            }
            totalChildVectors[i].resize(offsets[i][omp_get_num_threads()]);
        }
        // Update global
        #pragma omp barrier
        std::move(childVectors[0].begin(), childVectors[0].end(), totalChildVectors[0].begin()+offsets[0][omp_get_thread_num()]);
        std::move(childVectors[1].begin(), childVectors[1].end(), totalChildVectors[1].begin()+offsets[1][omp_get_thread_num()]);
        std::move(childVectors[2].begin(), childVectors[2].end(), totalChildVectors[2].begin()+offsets[2][omp_get_thread_num()]);
        std::move(childVectors[3].begin(), childVectors[3].end(), totalChildVectors[3].begin()+offsets[3][omp_get_thread_num()]);
        // for(int i = 0; i < 4; i++)
        // {
        //     int j = (omp_get_thread_num() + i)%4;
        //     if(childVectors[j].size())
        //     {
        //         #pragma omp critical
        //         {
        //             totalChildVectors[j].insert(totalChildVectors[j].end(), childVectors[j].begin(), childVectors[j].end());
        //         }
        //     }
        // }
      }
      delete[] offsets[0];
      delete[] offsets[1];
      delete[] offsets[2];
      delete[] offsets[3];
      
      #pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < 4; i++)
      {
        if (i == 0)
          curNode->children[0] = buildQuadTree(totalChildVectors[0], bmin, pivot);
        else if (i == 1)
        {
          Vec2 topRightMin = {pivot.x, bmin.y};
          Vec2 topRightMax = {bmax.x, pivot.y};
          curNode->children[1] = buildQuadTree(totalChildVectors[1], topRightMin, topRightMax);
        }
        else if (i == 2)
        {
          Vec2 bottomLeftMin = {bmin.x, pivot.y};
          Vec2 bottomLeftMax = {pivot.x, bmax.y};
          curNode->children[2] = buildQuadTree(totalChildVectors[2], bottomLeftMin, bottomLeftMax);
        }
        else
          curNode->children[3] = buildQuadTree(totalChildVectors[3], pivot, bmax);
      }

      return curNode;
    }

    return nullptr;
  }

  // Convert leaf into a branch
  std::unique_ptr<QuadTreeNode> makeBranch(QuadTreeNode *leaf, Vec2 bmin, Vec2 bmax)
  {
    // Set up branch
    std::unique_ptr<QuadTreeNode> curNode(new QuadTreeNode);
    curNode->isLeaf = false;
    // Bin particles into children
    Vec2 pivot = {(bmax.x + bmin.x) / 2, (bmax.y + bmin.y) / 2};
    std::vector<Particle> childVectors[4];
    for(int idx = 0; idx < leaf->particles.size(); idx++)
    {
        const Particle &p = leaf->particles[idx];
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
    // Make children
    for(int i = 0; i < 4; i++)
    {
      std::unique_ptr<QuadTreeNode> child(new QuadTreeNode);
      child->isLeaf = true;
      child->particles = childVectors[i];
      curNode->children[0] = std::move(child);
    }
    return curNode;
  }

  std::unique_ptr<QuadTreeNode> fuseTrees(
    QuadTreeNode *nodeA, 
    QuadTreeNode *nodeB,
    Vec2 bmin, Vec2 bmax) {
      std::unique_ptr<QuadTreeNode> curNode(new QuadTreeNode);
      if(nodeA->isLeaf && nodeB->isLeaf)
      {
        // Both nodes are leaves; combine them
        curNode->isLeaf = true;
        curNode->particles.reserve(nodeA->particles.size() + nodeB->particles.size());
        curNode->particles.insert(curNode->particles.end(), nodeA->particles.begin(), nodeA->particles.end());
        curNode->particles.insert(curNode->particles.end(), nodeB->particles.begin(), nodeB->particles.end());
        return curNode;
      }
      else if((nodeA->isLeaf && !nodeB->isLeaf) || (!nodeA->isLeaf && nodeB->isLeaf))
      {
        // One node is branch, other is leaf
        if(nodeA->isLeaf)
        {
          std::unique_ptr<QuadTreeNode> branched = makeBranch(nodeA, bmin, bmax);
          nodeA = branched.get();
          Vec2 pivot = {(bmax.x + bmin.x) / 2, (bmax.y + bmin.y) / 2};
          Vec2 topRightMin = {pivot.x, bmin.y};
          Vec2 topRightMax = {bmax.x, pivot.y};
          Vec2 bottomLeftMin = {bmin.x, pivot.y};
          Vec2 bottomLeftMax = {pivot.x, bmax.y};
          curNode->children[0] = fuseTrees(nodeA->children[0].get(), nodeB->children[0].get(), bmin, pivot);
          curNode->children[1] = fuseTrees(nodeA->children[1].get(), nodeB->children[1].get(), topRightMin, topRightMax);
          curNode->children[2] = fuseTrees(nodeA->children[2].get(), nodeB->children[2].get(), bottomLeftMin, bottomLeftMax);
          curNode->children[3] = fuseTrees(nodeA->children[3].get(), nodeB->children[3].get(), pivot, bmax);
          return curNode;
        }
        else
        {
          std::unique_ptr<QuadTreeNode> branched = makeBranch(nodeB, bmin, bmax);
          nodeB = branched.get();
          Vec2 pivot = {(bmax.x + bmin.x) / 2, (bmax.y + bmin.y) / 2};
          Vec2 topRightMin = {pivot.x, bmin.y};
          Vec2 topRightMax = {bmax.x, pivot.y};
          Vec2 bottomLeftMin = {bmin.x, pivot.y};
          Vec2 bottomLeftMax = {pivot.x, bmax.y};
          curNode->children[0] = fuseTrees(nodeA->children[0].get(), nodeB->children[0].get(), bmin, pivot);
          curNode->children[1] = fuseTrees(nodeA->children[1].get(), nodeB->children[1].get(), topRightMin, topRightMax);
          curNode->children[2] = fuseTrees(nodeA->children[2].get(), nodeB->children[2].get(), bottomLeftMin, bottomLeftMax);
          curNode->children[3] = fuseTrees(nodeA->children[3].get(), nodeB->children[3].get(), pivot, bmax);
          return curNode;
        }
      }
      else
      {
        // Both nodes are branches; recur
        Vec2 pivot = {(bmax.x + bmin.x) / 2, (bmax.y + bmin.y) / 2};
        Vec2 topRightMin = {pivot.x, bmin.y};
        Vec2 topRightMax = {bmax.x, pivot.y};
        Vec2 bottomLeftMin = {bmin.x, pivot.y};
        Vec2 bottomLeftMax = {pivot.x, bmax.y};
        curNode->children[0] = fuseTrees(nodeA->children[0].get(), nodeB->children[0].get(), bmin, pivot);
        curNode->children[1] = fuseTrees(nodeA->children[1].get(), nodeB->children[1].get(), topRightMin, topRightMax);
        curNode->children[2] = fuseTrees(nodeA->children[2].get(), nodeB->children[2].get(), bottomLeftMin, bottomLeftMax);
        curNode->children[3] = fuseTrees(nodeA->children[3].get(), nodeB->children[3].get(), pivot, bmax);
        return curNode;
      }
  }

  std::unique_ptr<QuadTreeNode> buildQuadTreeSuper(std::vector<Particle> &particles,
                                              Vec2 bmin, Vec2 bmax) {
    
    // A and B are constructed from particles; C is their fusion
    auto quadTreeA = std::make_unique<QuadTree>();
    auto quadTreeB = std::make_unique<QuadTree>();
    // Set bounds
    quadTreeA->bmin = bmin;
    quadTreeA->bmax = bmax;
    quadTreeB->bmin = bmin;
    quadTreeB->bmax = bmax;
    // Set up particles
    std::vector<Particle> particlesA(particles.begin(), particles.begin()+particles.size()/2);
    std::vector<Particle> particlesB(particles.begin()+particles.size()/2, particles.end());
    // Build both trees
    quadTreeA->root = buildQuadTree(particlesA, bmin, bmax);
    quadTreeA->root = buildQuadTree(particlesB, bmin, bmax);
    // Return fused tree
    return fuseTrees(quadTreeA->root.get(), quadTreeB->root.get(), bmin, bmax);
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
    quadTree->root = buildQuadTreeSuper(particles, bmin, bmax);
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
