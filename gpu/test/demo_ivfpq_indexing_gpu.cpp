/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
//#include <sys/time.h>

#include "../../Clustering.h"
#include "../../utils.h"
#include "../GpuIndexFlat.h"
#include "../StandardGpuResources.h"
#include "../GpuIndexIVFPQ.h"

#include "../GpuAutoTune.h"
#include "../../index_io.h"

#include "opencvincludes.h"

double elapsed ()
{
    /*struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;*/

	return cv::getTickCount() * 1.0 / cv::getTickFrequency() ;
}
void gpuKmeansDemo();

int main ()
{
	//gpuKmeansDemo();

	//return 0;

	double t0 = cv::getTickCount(); //elapsed();

    // dimension of the vectors to index
    int d = 128;

    // size of the database we plan to index
    size_t nb = 200 * 1000;

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 100 * 1000;

    int dev_no = 0;
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors
    int ncentroids = int (4 * sqrt (nb));

    faiss::gpu::StandardGpuResources resources;


    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = dev_no;

    faiss::gpu::GpuIndexIVFPQ index (
      &resources, d, ncentroids, 4, 8, faiss::METRIC_L2, config);

    { // training
        printf ("[%.3f s] Generating %ld vectors in %dD for training\n",
                elapsed() - t0, nt, d);

        std::vector <float> trainvecs (nt * d);
        for (size_t i = 0; i < nt * d; i++) {
			trainvecs[i] = (rand() / (RAND_MAX + 1.0));//drand48();
        }

        printf ("[%.3f s] Training the index\n",
                elapsed() - t0);
        index.verbose = true;

        index.train (nt, trainvecs.data());
    }

    { // I/O demo
		const char *outfilename = "I:\\Data\\Temp\\index_trained.faissindex";//"/tmp/index_trained.faissindex";
        printf ("[%.3f s] storing the pre-trained index to %s\n",
                elapsed() - t0, outfilename);

        faiss::Index * cpu_index = faiss::gpu::index_gpu_to_cpu (&index);

        write_index (cpu_index, outfilename);

        delete cpu_index;
    }

    size_t nq;
    std::vector<float> queries;

    { // populating the database
        printf ("[%.3f s] Building a dataset of %ld vectors to index\n",
                elapsed() - t0, nb);

        std::vector <float> database (nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = (rand() / (RAND_MAX + 1.0));//drand48();drand48();
        }

        printf ("[%.3f s] Adding the vectors to the index\n",
                elapsed() - t0);

        index.add (nb, database.data());

        printf ("[%.3f s] done\n", elapsed() - t0);

        // remember a few elements from the database as queries
        int i0 = 1234;
        int i1 = 1243;

        nq = i1 - i0;
        queries.resize (nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries [(i - i0) * d  + j]  = database [i * d + j];
            }
        }

    }

    { // searching the database
        int k = 5;
        printf ("[%.3f s] Searching the %d nearest neighbors "
                "of %ld vectors in the index\n",
                elapsed() - t0, k, nq);

        std::vector<int64_t> nns (k * nq);
        std::vector<float>               dis (k * nq);

        index.search (nq, queries.data(), k, dis.data(), nns.data());

        printf ("[%.3f s] Query results (vector ids, then distances):\n",
                elapsed() - t0);

        for (int i = 0; i < nq; i++) {
            printf ("query %2d: ", i);
            for (int j = 0; j < k; j++) {
                printf ("%7ld ", nns[j + i * k]);
            }
            printf ("\n     dis: ");
            for (int j = 0; j < k; j++) {
                printf ("%7g ", dis[j + i * k]);
            }
            printf ("\n");
        }

        printf ("note that the nearest neighbor is not at "
                "distance 0 due to quantization errors\n");
    }

    return 0;
}

// just generate some random vecs in a hypercube (CPU)
std::vector<float> makeRandomVecs(size_t numVecs, int dim) {
	std::vector<float> vecs(numVecs * dim);
	faiss::float_rand(vecs.data(), vecs.size(), 1);
	return vecs;
}


void gpuKmeansDemo()
{
	// Reserves 18% of GPU memory for temporary work by default; the
	// size can be adjusted, or your own implementation of GpuResources
	// can be made to manage memory in a different way.
	faiss::gpu::StandardGpuResources res;

	int dim = 128;
	int numberOfEMIterations = 20;
	size_t numberOfClusters = 16384;//20000;
	size_t numVecsToCluster = 32 * 1024 * 1024; //5000000;

	// generate a bunch of random vectors; note that this is on the CPU!
	std::vector<float> vecs = makeRandomVecs(numVecsToCluster, dim);
	faiss::gpu::GpuIndexFlatConfig config;
	config.device = 0;            // this is the default
	config.useFloat16 = false;    // this is the default
	faiss::gpu::GpuIndexFlatL2 index(&res, dim, config);

	faiss::ClusteringParameters cp;
	cp.niter = numberOfEMIterations;
	cp.verbose = true; // print out per-iteration stats

					   // For spherical k-means, use GpuIndexFlatIP and set cp.spherical = true

					   // By default faiss only samples 256 vectors per centroid, in case
					   // you are asking for too few centroids for too many vectors.
					   // e.g., numberOfClusters = 1000, numVecsToCluster = 1000000 would
					   // only sample 256000 vectors.
					   //
					   // You can override this to use any number of clusters
					   // cp.max_points_per_centroid =
					   //   ((numVecsToCluster + numberOfClusters - 1) / numberOfClusters);

	faiss::Clustering kMeans(dim, numberOfClusters, cp);

	// do the work!
	kMeans.train(numVecsToCluster, vecs.data(), index);

	// kMeans.centroids contains the resulting cluster centroids (on CPU)
	printf("centroid 3 dim 6 is %f\n", kMeans.centroids[3 * dim + 6]);
}