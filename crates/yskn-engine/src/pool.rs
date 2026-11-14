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

/// Row ranges of one or more matrices that read the same input, claimed task by task.
struct KernelJob {
    matrices: Arc<[Matrix]>,
    /// (matrix index, output rows) per task.
    tasks: Arc<[(usize, Range<usize>)]>,
    activation: Arc<[i8]>,
    scales: Arc<[i32]>,
    next: Arc<AtomicUsize>,
    kernel: Kernel,
    reply: mpsc::Sender<(usize, Result<Vec<i64>>)>,
}

enum Job {
    Kernel(KernelJob),
    Range(Box<dyn FnOnce() + Send>),
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

    /// One worker per physical core of every performance level, read from this
    /// machine ([`crate::Machine::default_threads`]; 10 on the 4P+6E M5, D10).
    pub fn default_threads() -> usize {
        static THREADS: OnceLock<usize> = OnceLock::new();
        *THREADS.get_or_init(|| crate::Machine::detect().default_threads())
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
                        let job = match job {
                            Job::Range(run) => {
                                run();
                                continue;
                            }
                            Job::Kernel(job) => job,
                        };
                        loop {
                            let task = job.next.fetch_add(1, Ordering::Relaxed);
                            let Some((matrix, rows)) = job.tasks.get(task) else {
                                break;
                            };
                            let result = kernel::compute(
                                &job.matrices[*matrix],
                                &job.activation,
                                &job.scales,
                                rows.clone(),
                                job.kernel,
                            );
                            if job.reply.send((task, result)).is_err() {
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

    /// Run independent indices on the same persistent workers, returning index order.
    /// Each worker owns its results; the caller's closure may borrow immutable data.
    pub fn map_range<T: Send, F: Fn(usize) -> Result<T> + Sync>(
        &mut self,
        range: Range<usize>,
        function: F,
    ) -> Result<Vec<T>> {
        if range.start > range.end {
            return Err(Error::InvalidInput("reversed index range".into()));
        }
        if range.is_empty() {
            return Ok(Vec::new());
        }
        let mut output: Vec<Option<T>> = Vec::new();
        output
            .try_reserve_exact(range.len())
            .map_err(|_| Error::InvalidInput("range output allocation failed".into()))?;
        output.resize_with(range.len(), || None);
        let count = self.threads().min(range.len());
        let next = AtomicUsize::new(0);
        let (reply, receive) = mpsc::channel();
        let mut sent = 0;
        let mut failure = None;
        for sender in self.senders.iter().take(count) {
            let function = &function;
            let next = &next;
            let reply = reply.clone();
            let length = range.len();
            let first = range.start;
            let run: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut results = Vec::new();
                    loop {
                        let offset = next.fetch_add(1, Ordering::Relaxed);
                        if offset >= length {
                            break;
                        }
                        let index = first + offset;
                        // Caught per index so a panic is reported at its own index.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            function(index)
                        }))
                        .unwrap_or_else(|_| {
                            Err(Error::InvalidInput("range worker panicked".into()))
                        });
                        results.push((index, result));
                    }
                    results
                }))
                .map_err(|_| Error::InvalidInput("range worker panicked".into()));
                let _ = reply.send(result);
            });
            // SAFETY: function and next stay alive until all `sent` jobs reply below,
            // including on dispatch/numeric errors or closure panic. Each job catches
            // unwinding and sends owned results after its last use of the borrows.
            // No borrowed result escapes: T retains its original caller-visible lifetime.
            let run: Box<dyn FnOnce() + Send> = unsafe { std::mem::transmute(run) };
            if sender.send(Job::Range(run)).is_err() {
                failure = Some((
                    usize::MAX,
                    Error::InvalidInput("range worker disconnected".into()),
                ));
                break;
            }
            sent += 1;
        }
        drop(reply);
        // Like a sequential loop, report the failing index that comes first;
        // failures without an index rank after all indexed ones.
        let mut fail = |index: usize, error: Error| {
            if failure.as_ref().is_none_or(|(first, _)| index < *first) {
                failure = Some((index, error));
            }
        };
        for _ in 0..sent {
            match receive.recv() {
                Ok(Ok(results)) => {
                    for (index, result) in results {
                        match result {
                            Ok(value) => output[index - range.start] = Some(value),
                            Err(error) => fail(index, error),
                        }
                    }
                }
                Ok(Err(error)) => fail(usize::MAX, error),
                Err(_) => fail(
                    usize::MAX,
                    Error::InvalidInput("range worker failed".into()),
                ),
            }
        }
        if let Some((_, error)) = failure {
            return Err(error);
        }
        output
            .into_iter()
            .map(|v| v.ok_or_else(|| Error::InvalidInput("range result missing".into())))
            .collect()
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
        if n == 0 || n > matrix.descriptor().physical_rows {
            return Err(Error::InvalidInput(
                "invalid activation width or output row count".into(),
            ));
        }
        let [sums] = self
            .weighted_sums_rows(&[(matrix, 0..n)], activation, scales)?
            .try_into()
            .expect("one part");
        Ok(sums)
    }

    /// Row-major M×len sums of each (matrix, physical row range), all reading the
    /// same input, computed by one batch of worker tasks. Empty ranges are allowed.
    pub fn weighted_sums_rows(
        &mut self,
        parts: &[(&Matrix, Range<usize>)],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        let mut outputs = Vec::with_capacity(parts.len());
        let mut tasks = Vec::new();
        for (index, (matrix, rows)) in parts.iter().enumerate() {
            let k = matrix.descriptor().cols;
            if activation.is_empty()
                || !activation.len().is_multiple_of(k)
                || rows.start > rows.end
                || rows.end > matrix.descriptor().physical_rows
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
                .checked_mul(rows.len())
                .ok_or_else(|| Error::InvalidInput("output length overflow".into()))?;
            let mut output = Vec::new();
            output
                .try_reserve_exact(output_len)
                .map_err(|_| Error::InvalidInput("output allocation failed".into()))?;
            output.resize(output_len, 0);
            outputs.push(output);
            // Small ranges still expose one tile per worker. Large ones use
            // 256-row jobs to amortize channels while balancing heterogeneous cores.
            let task_rows = if rows.len() / self.threads() < 256 {
                32
            } else {
                256
            };
            tasks.extend(
                rows.clone()
                    .step_by(task_rows)
                    .map(|start| (index, start..(start + task_rows).min(rows.end))),
            );
        }
        if tasks.is_empty() {
            return Ok(outputs);
        }
        let matrices: Arc<[Matrix]> = parts.iter().map(|(m, _)| (*m).clone()).collect();
        let tasks: Arc<[(usize, Range<usize>)]> = tasks.into();
        let activation: Arc<[i8]> = activation.into();
        let scales: Arc<[i32]> = scales.into();
        let (reply, receive) = mpsc::channel();
        let count = self.threads().min(tasks.len());
        let next = Arc::new(AtomicUsize::new(0));
        for worker in 0..count {
            self.senders[worker]
                .send(Job::Kernel(KernelJob {
                    matrices: Arc::clone(&matrices),
                    tasks: Arc::clone(&tasks),
                    activation: Arc::clone(&activation),
                    scales: Arc::clone(&scales),
                    next: Arc::clone(&next),
                    kernel: self.kernel,
                    reply: reply.clone(),
                }))
                .map_err(|_| Error::InvalidInput("kernel worker disconnected".into()))?;
        }
        drop(reply);
        let mut failure = None;
        // Drain all replies even after a numeric error, so a failed call is fully finished.
        for _ in 0..tasks.len() {
            let (task, result) = receive
                .recv()
                .map_err(|_| Error::InvalidInput("kernel worker failed".into()))?;
            match result {
                Ok(values) => {
                    let (part, rows) = &tasks[task];
                    let (output, first) = (&mut outputs[*part], parts[*part].1.start);
                    let (n, width) = (parts[*part].1.len(), rows.len());
                    for input in 0..values.len() / width.max(1) {
                        let start = input * n + rows.start - first;
                        output[start..start + width]
                            .copy_from_slice(&values[input * width..(input + 1) * width]);
                    }
                }
                Err(error) => failure = Some(error),
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(outputs),
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
