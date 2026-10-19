# Dataset provenance and acquisition

The original `Readme.txt` links to UNSW ToN-IoT, UNSW BoT-IoT, and UNB CSE-CIC-IDS2018 dataset family pages. These original releases are not interchangeable with the standardized NetFlow-v2 CSVs. Revised configurations explicitly require v2 columns.

Authoritative index: https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets

| Dataset | Official UQ delivery page | Published rows (reference only) |
|---|---|---:|
| NF-ToN-IoT-v2 | https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5 | 16,940,496 |
| NF-BoT-IoT-v2 | https://rdm.uq.edu.au/files/b97fa3c0-ef9c-11ed-8007-832d3bc76ba8 | 37,763,497 |
| NF-CSE-CIC-IDS2018-v2 | https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848 | 18,893,708 |

The UQ page describes 43 extended NetFlow features and academic-use/citation requirements. Reference: Mohanad Sarhan, Siamak Layeghy, Marius Portmann, *Towards a Standard Feature Set for Network Intrusion Detection System Datasets*, DOI https://doi.org/10.1007/s11036-021-01843-0 . Cite the original dataset creators as appropriate too.

The delivery pages are JavaScript applications; downloading their HTML is **not** downloading data. Use their Download File control and verify the actual resulting archive/CSV. Do not commit raw records to GitHub. Record source URL, retrieval time, archive and extracted file SHA-256, original header, unmodified label frequencies, and any extraction/sampling operations. Published source counts above are not experimental counts.

Put files under ignored `data/` and update the CSV configuration. Never manufacture time columns. If the CSV lacks timestamps, temporal leakage cannot be excluded by a claimed chronological split. Inspect original `Attack` label strings before accepting a multiclass label map. The initial adapters reject unknown labels rather than discarding them.
