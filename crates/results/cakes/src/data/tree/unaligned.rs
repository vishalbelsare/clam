//! Data of unaligned sequences.

use std::collections::HashMap;

use abd_clam::{
    cakes::{KnnBreadthFirst, KnnDepthFirst, KnnRepeatedRnn, ParSearchAlgorithm, RnnClustered},
    cluster::{adapter::ParBallAdapter, ClusterIO, Csv, ParPartition},
    dataset::{AssociatesMetadata, AssociatesMetadataMut, ParDatasetIO},
    metric::Levenshtein,
    pancakes::{CodecData, SquishyBall},
    Ball, Cluster, Dataset, FlatVec,
};
use distances::Number;

use super::{instances::Unaligned, PathManager};

type I = Unaligned;
type U = u32;
type M = String;
type Co = FlatVec<I, M>;
type B = Ball<U>;
type Dec = CodecData<I, M>;
type Sb = SquishyBall<U, B>;
type Hits = Vec<Vec<(usize, U)>>;

/// The group of types used for the datasets of unaligned sequences.
pub struct Group {
    path_manager: PathManager,
    uncompressed: Co,
    ball: B,
    compressed: Dec,
    squishy_ball: Sb,
    #[allow(dead_code)]
    query_ids: Vec<M>,
    queries: Vec<I>,
}

impl Group {
    /// Creates a new group of datasets and trees for benchmarks with unaligned sequences.
    ///
    /// # Errors
    ///
    /// - If there is an error deserializing or serializing the data.
    /// - If there is an error reading/writing serialized data to/from disk.
    /// - If there is an error writing the trees to csv files.
    pub fn new(path_manager: PathManager, uncompressed: Co, queries: Vec<(M, I)>) -> Result<Self, String> {
        let metric = Levenshtein;
        let query_path = path_manager.queries_path();
        if !query_path.exists() {
            // Serialize the queries to disk.
            let bytes = bitcode::encode(&queries).map_err(|e| e.to_string())?;
            std::fs::write(&query_path, &bytes).map_err(|e| e.to_string())?;
        }
        let (query_ids, queries) = queries.into_iter().unzip();

        let ball_path = path_manager.ball_path();
        let ball = if ball_path.exists() {
            ftlog::info!("Reading ball from {ball_path:?}");
            B::read_from(&ball_path)?
        } else {
            // Create the ball from scratch.
            ftlog::info!("Building ball.");
            let mut depth = 0;
            let seed = Some(42);

            let indices = (0..uncompressed.cardinality()).collect::<Vec<_>>();
            let mut ball = Ball::par_new(&uncompressed, &metric, &indices, 0, seed)
                .unwrap_or_else(|e| unreachable!("We ensured that indices is non-empty: {e}"));
            let depth_delta = abd_clam::utils::max_recursion_depth();

            let criteria = |c: &B| c.depth() < 1;
            ball.par_partition(&uncompressed, &metric, &criteria, seed);

            while ball.leaves().into_iter().any(|c| !c.is_singleton()) {
                depth += depth_delta;
                let criteria = |c: &B| c.depth() < depth;
                ball.par_partition_further(&uncompressed, &metric, &criteria, seed);
            }

            let num_leaves = ball.leaves().len();
            ftlog::info!("Built ball with {num_leaves} leaves.");

            // Serialize the ball to disk.
            ftlog::info!("Writing ball to {ball_path:?}");
            ball.write_to(&ball_path)?;

            // Write the ball to a CSV file.
            let csv_path = path_manager.ball_csv_path();
            ftlog::info!("Writing ball to CSV at {csv_path:?}");
            ball.write_to_csv(&csv_path)?;

            ball
        };

        let squishy_ball_path = path_manager.squishy_ball_path();
        let compressed_path = path_manager.compressed_path();

        let (squishy_ball, compressed) = if squishy_ball_path.exists() && compressed_path.exists() {
            ftlog::info!("Reading squishy ball from {squishy_ball_path:?}");
            let squishy_ball = Sb::read_from(&squishy_ball_path)?;

            ftlog::info!("Reading compressed data from {compressed_path:?}");
            let codec_data = Dec::par_read_from(&compressed_path)?;

            (squishy_ball, codec_data)
        } else {
            ftlog::info!("Building squishy ball and compressed data.");
            let (squishy_ball, codec_data) = {
                let (squishy_ball, data) = SquishyBall::par_from_ball_tree(ball.clone(), uncompressed.clone(), &metric);

                // Write it to a CSV file.
                squishy_ball.write_to_csv(&path_manager.squishy_ball_csv_path())?;

                (squishy_ball, data.with_metadata(uncompressed.metadata())?)
            };

            let num_leaves = squishy_ball.leaves().len();
            ftlog::info!("Built squishy ball with {num_leaves} leaves.");

            let num_bytes = codec_data
                .leaf_bytes()
                .iter()
                .map(|(_, bytes)| core::mem::size_of::<usize>() + bytes.len())
                .sum::<usize>();
            ftlog::info!("Built compressed dataset with {num_bytes} leaf bytes.");

            // Serialize the squishy ball and the compressed dataset to disk.
            ftlog::info!("Writing squishy ball to {squishy_ball_path:?}");
            squishy_ball.write_to(&squishy_ball_path)?;

            ftlog::info!("Writing compressed data to {compressed_path:?}");
            codec_data.par_write_to(&compressed_path)?;

            (squishy_ball, codec_data)
        };

        Ok(Self {
            path_manager,
            uncompressed,
            ball,
            compressed,
            squishy_ball,
            query_ids,
            queries,
        })
    }

    fn bench_search<Aco, Adec>(&self, num_queries: usize, alg_a: &Aco, alg_b: &Adec) -> Result<Vec<String>, String>
    where
        Aco: ParSearchAlgorithm<I, U, B, Levenshtein, Co>,
        Adec: ParSearchAlgorithm<I, U, Sb, Levenshtein, Dec>,
    {
        let metric = &Levenshtein;
        let name = alg_a.name();

        let queries = &self.queries[..num_queries];
        ftlog::info!("Running benchmarks for compressive search on {num_queries} queries with {name}");

        let uncompressed_start = std::time::Instant::now();
        let uncompressed_hits = alg_a.par_batch_search(&self.uncompressed, metric, &self.ball, queries);
        let uncompressed_time = uncompressed_start.elapsed().as_secs_f32() / num_queries.as_f32();
        ftlog::info!(
            "Algorithm {name} took {uncompressed_time:.3e} seconds per query uncompressed time on {}",
            self.path_manager.name()
        );

        let compressed_start = std::time::Instant::now();
        let compressed_hits = alg_b.par_batch_search(&self.compressed, metric, &self.squishy_ball, queries);
        let compressed_time = compressed_start.elapsed().as_secs_f32() / num_queries.as_f32();
        ftlog::info!(
            "Algorithm {name} took {compressed_time:.3e} seconds per query compressed time on {}",
            self.path_manager.name()
        );

        self.verify_hits(uncompressed_hits, compressed_hits)?;

        let slowdown = compressed_time / uncompressed_time;
        Ok(vec![
            format!("uncompressed: {uncompressed_time:.4e}"),
            format!("uncompressed_throughput: {:.4e}", 1.0 / uncompressed_time),
            format!("compressed: {compressed_time:.4e}"),
            format!("compressed_throughput: {:.4e}", 1.0 / compressed_time),
            format!("slowdown: {slowdown:.4}"),
        ])
    }

    /// Run benchmarks for compressive search on the dataset.
    ///
    /// # Errors
    ///
    /// - If there is an error writing the times to disk.
    pub fn bench_compressive_search(&self, num_queries: usize) -> Result<(), String> {
        let num_queries = Ord::min(num_queries, self.queries.len());
        ftlog::info!("Running benchmarks for compressive search on {num_queries} queries.");

        let mut times = HashMap::new();
        for radius in [5, 10, 100] {
            let times_inner = self.bench_search(num_queries, &RnnClustered(radius), &RnnClustered(radius))?;
            times.insert(format!("RnnClustered({radius})"), times_inner);
        }

        for k in [1, 10, 100] {
            let times_inner = self.bench_search(num_queries, &KnnRepeatedRnn(k, 2), &KnnRepeatedRnn(k, 2))?;
            times.insert(format!("KnnRepeatedRnn({k}, 2)"), times_inner);

            let times_inner = self.bench_search(num_queries, &KnnBreadthFirst(k), &KnnBreadthFirst(k))?;
            times.insert(format!("KnnBreadthFirst({k})"), times_inner);

            let times_inner = self.bench_search(num_queries, &KnnDepthFirst(k), &KnnDepthFirst(k))?;
            times.insert(format!("KnnDepthFirst({k})"), times_inner);
        }

        ftlog::info!("Writing times to disk.");
        serde_json::to_writer_pretty(
            std::fs::File::create(self.path_manager.times_path()).map_err(|e| e.to_string())?,
            &times,
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Checks that the hits from the uncompressed and compressed datasets are the same.
    #[allow(
        dead_code,
        unused_variables,
        clippy::unnecessary_wraps,
        clippy::needless_pass_by_value,
        clippy::unused_self
    )]
    fn verify_hits(&self, uncompressed: Hits, compressed: Hits) -> Result<(), String> {
        ftlog::warn!("Hit verification not yet implemented.");
        Ok(())
    }
}
