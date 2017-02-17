<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png">
  <img src="https://www.open-mpi.org/images/open-mpi-logo.png">
  <br><br>
</div>
-----------------

# TensorFlow with MPI

This repository contains a patched version of TensorFlow 0.12.1 which includes
the `tensorflow.contrib.mpi` namespace with MPI operations, including a
potentially CUDA-aware ring allreduce.

## Installation

Using this requires building TensorFlow from source with a CUDA-aware MPI of
your choice, and has been tested with [OpenMPI](https://www.open-mpi.org/)
integrated with [SLURM](https://slurm.schedmd.com/).

Install by following the [TensorFlow source installation instructions](https://www.tensorflow.org/install/install_sources). 
When you run `configure`, you will be prompted for whether you would like to
build TensorFlow with MPI, and, if so, what path your MPI installation is at.

Although it has only been tested with SLURM-integrated OpenMPI, it should also
work with any other CUDA-aware MPI implementation.

## Usage

The auto-generated documentation for TensorFlow includes usage examples. In
addition, we include a TensorFlow language model that we use for benchmarking
the allreduce in a real-world situation. In order to run the language model
training, make sure you `pip install -r allreduce-requirements.txt` to install
all Python dependencies.

After that, you should be able to run `allreduce-test.py` with the appropriate
training and validation datasets and vocabulary. We train on the Billion Words dataset, which 
is a text file with one sentence per line, as follows:

```
...
To Mo concerning the food log you kept -- Dr. Buchholz recommends the same thing .
The CBO estimates that only 23 percent of that would be spent in 2009 and 2010 .
Even so , Democrats slammed Bush as out of touch .
An information campaign will be launched later to raise awareness of employment rights and how to enforce them .
...
```

The vocabulary file is a list of the top most common vocabulary words:

```
<unk>
the
,
.
to
of
and
a
in
"
's
that
for
on
is
The
was
with
said
as
at
...
```

You should be able to run training with a command as follows:

```bash
# If you have SLURM with a CUDA-aware MPI integrated, you can use `srun` to
# launch your job. Otherwise, you will need to use `mpirun` and appropriately
# set `CUDA_VISIBLE_DEVICES` to choose which GPUs to use.
srun --partition=K40x4 --ntasks=4 --gres=gpu:4 \
    python allreduce-test.py \
        --train-data train.txt \
        --validation-data train.txt \
        --vocab vocab.txt \
        --vocab-size 10000 \
        --batch-size 32 \
        --max-iterations 10000
```

## Support

We do not offer any sort of official support or maintenance for this patch.
However, if you would like to use it and run into trouble, feel free to [file a Github issue](https://github.com/baidu-research/tensorflow-allreduce/issues)
and we may be able to help.
