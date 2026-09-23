# Measurement record: the end-to-end decode matrix

The raw JSON in this directory is git-ignored; this file is the committed record of the
measurement. It is the output of

    python -m bench.render_table results/e2e_decode.json

over `results/e2e_decode.json` as collected by `bench/run_e2e_matrix.sh` with
`REPEATS=2` and a 20 s cool-down: 48 runs (the short and novel cells at git `ee87f75`,
the code-corpus cells at git `8adca11`), one configuration per process, sequential, on
this machine. Each cell below is the latest of the two passes; the technical writeup on
our website quotes the average of the two passes instead. Regenerate with the command
above.

End-to-end decode matrix (Apple M5 Pro, MLX 0.32.2), 48 runs.

### chat (27 prompt tokens)
| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |
|---|---|---|---|---|---|
| dense | 344 | 107.3 | 9.32 | - | 17.9 |
| sparse | 419 | 123.3 | 8.11 | - | 18.1 |
| dense + MTP 2 | 349 | 121.1 | 8.25 | 1.30 of 2 (hist [9, 10, 21]) | 18.7 |
| sparse + MTP 2 | 420 | 138.3 | 7.23 | 1.26 of 2 (hist [10, 11, 21]) | 18.9 |

### code (50 prompt tokens)
| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |
|---|---|---|---|---|---|
| dense | 356 | 103.7 | 9.64 | - | 18.0 |
| sparse | 483 | 116.9 | 8.55 | - | 18.2 |
| dense + MTP 2 | 387 | 134.8 | 7.42 | 1.75 of 2 (hist [6, 11, 76]) | 18.7 |
| sparse + MTP 2 | 517 | 166.5 | 6.01 | 1.68 of 2 (hist [10, 10, 75]) | 19.0 |

### moby_dick.txt (2048 prompt tokens)
| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |
|---|---|---|---|---|---|
| dense | 1920 | 96.3 | 10.39 | - | 21.3 |
| sparse | 2005 | 113.6 | 8.80 | - | 21.3 |
| dense + MTP 2 | 1941 | 91.3 | 10.95 | 0.98 of 2 (hist [25, 15, 24]) | 22.2 |
| sparse + MTP 2 | 2104 | 105.6 | 9.47 | 0.84 of 2 (hist [33, 15, 22]) | 22.2 |

### moby_dick.txt (8192 prompt tokens)
| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |
|---|---|---|---|---|---|
| dense | 1907 | 94.4 | 10.59 | - | 21.4 |
| sparse | 1942 | 109.3 | 9.15 | - | 21.4 |
| dense + MTP 2 | 1927 | 61.7 | 16.20 | 0.48 of 2 (hist [56, 19, 11]) | 22.3 |
| sparse + MTP 2 | 1932 | 70.0 | 14.29 | 0.46 of 2 (hist [54, 26, 7]) | 22.3 |

### code_corpus.txt (2048 prompt tokens)
| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |
|---|---|---|---|---|---|
| dense | 1986 | 100.8 | 9.92 | - | 21.3 |
| sparse | 2036 | 114.7 | 8.72 | - | 21.3 |
| dense + MTP 2 | 1951 | 101.3 | 9.87 | 1.19 of 2 (hist [19, 9, 30]) | 22.2 |
| sparse + MTP 2 | 2027 | 120.2 | 8.32 | 1.15 of 2 (hist [19, 12, 28]) | 22.2 |

### code_corpus.txt (8192 prompt tokens)
| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |
|---|---|---|---|---|---|
| dense | 1926 | 96.9 | 10.32 | - | 21.4 |
| sparse | 1998 | 110.8 | 9.02 | - | 21.4 |
| dense + MTP 2 | 1930 | 89.3 | 11.20 | 1.33 of 2 (hist [12, 13, 30]) | 22.3 |
| sparse + MTP 2 | 1997 | 108.8 | 9.19 | 1.37 of 2 (hist [11, 12, 31]) | 22.3 |
