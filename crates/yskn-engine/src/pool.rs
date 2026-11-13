use crate::{Error, Kernel, Matrix, Result, kernel};
use std::{
    ops::Range,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

struct Job {
    matrix: Matrix,
    activation: Arc<[i8]>,
    scales: Arc<[i32]>,
    next: Arc<AtomicUsize>,
    rows: usize,
    task_rows: usize,
    kernel: Kernel,
    reply: mpsc::Sender<(Range<usize>, Result<Vec<i64>>)>,
}

/// Persistent workers. Calls borrow mutably so one engine executes one call at a time.
pub struct Engine {
    kernel: Kernel,
    senders: Vec<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl Engine {
    /// Read YSKN_KERNEL once per process. Explicit `new` never reads the environment.
    pub fn from_env() -> Result<Self> {
        static CHOICE: OnceLock<Option<String>> = OnceLock::new();
        let choice = CHOICE.get_or_init(|| std::env::var("YSKN_KERNEL").ok());
        let kernel = match choice {
            Some(s) => s.parse()?,
            None => Kernel::fastest(),
        };
        Self::new(kernel, Self::default_threads())
    }

    /// Measured on the 4P+6E M5: cap at ten, respecting smaller available CPU counts.
    pub fn default_threads() -> usize {
        thread::available_parallelism().map_or(1, |n| n.get().min(10))
    }

    pub fn new(kernel: Kernel, threads: usize) -> Result<Self> {
        if !kernel.available() {
            return Err(Error::InvalidInput(format!(
                "{} unavailable on this CPU",
                kernel.name()
            )));
        }
        if threads == 0 {
            return Err(Error::InvalidInput("thread count must be positive".into()));
        }
        let mut engine = Self {
            kernel,
            senders: Vec::new(),
            workers: Vec::new(),
        };
        for index in 0..threads {
            let (sender, receiver) = mpsc::channel::<Job>();
            let worker = thread::Builder::new()
                .name(format!("yskn-{index}"))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        loop {
                            let start = job.next.fetch_add(job.task_rows, Ordering::Relaxed);
                            if start >= job.rows {
                                break;
                            }
                            let rows = start..(start + job.task_rows).min(job.rows);
                            let result = kernel::compute(
                                &job.matrix,
                                &job.activation,
                                &job.scales,
                                rows.clone(),
                                job.kernel,
                            );
                            if job.reply.send((rows, result)).is_err() {
                                break;
                            }
                        }
                    }
                })?;
            engine.senders.push(sender);
            engine.workers.push(worker);
        }
        Ok(engine)
    }
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Row-major M×N sums, including padded physical output rows by default.
    pub fn weighted_sums(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<i64>> {
        self.weighted_sums_prefix(
            matrix,
            activation,
            scales,
            matrix.descriptor().physical_rows,
        )
    }

    /// Compute the first N physical rows, e.g. a vocabulary-head verification slice.
    pub fn weighted_sums_prefix(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
        n: usize,
    ) -> Result<Vec<i64>> {
        let k = matrix.descriptor().cols;
        if activation.is_empty()
            || !activation.len().is_multiple_of(k)
            || n == 0
            || n > matrix.descriptor().physical_rows
        {
            return Err(Error::InvalidInput(
                "invalid activation width or output row count".into(),
            ));
        }
        let m = activation.len() / k;
        if scales.len() != activation.len() / 512 {
            return Err(Error::InvalidInput(
                "integer scales must have M times K/512 elements".into(),
            ));
        }
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| Error::InvalidInput("output length overflow".into()))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| Error::InvalidInput("output allocation failed".into()))?;
        output.resize(output_len, 0);
        let activation: Arc<[i8]> = activation.into();
        let scales: Arc<[i32]> = scales.into();
        let (reply, receive) = mpsc::channel();
        // Small matrices still expose one tile per worker. Large ones use
        // 256-row jobs to amortize channels while balancing heterogeneous cores.
        let task_rows = if n / self.threads() < 256 { 32 } else { 256 };
        let tasks = n.div_ceil(task_rows);
        let count = self.threads().min(tasks);
        let next = Arc::new(AtomicUsize::new(0));
        for worker in 0..count {
            self.senders[worker]
                .send(Job {
                    matrix: matrix.clone(),
                    activation: Arc::clone(&activation),
                    scales: Arc::clone(&scales),
                    next: Arc::clone(&next),
                    rows: n,
                    task_rows,
                    kernel: self.kernel,
                    reply: reply.clone(),
                })
                .map_err(|_| Error::InvalidInput("kernel worker disconnected".into()))?;
        }
        drop(reply);
        let mut failure = None;
        // Drain all replies even after a numeric error, so a failed call is fully finished.
        for _ in 0..tasks {
            let (rows, result) = receive
                .recv()
                .map_err(|_| Error::InvalidInput("kernel worker failed".into()))?;
            match result {
                Ok(values) => {
                    for input in 0..m {
                        output[input * n + rows.start..input * n + rows.end]
                            .copy_from_slice(&values[input * rows.len()..(input + 1) * rows.len()]);
                    }
                }
                Err(error) => failure = Some(error),
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(output),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
