PYTHON ?= python3
SAMPLE_PYTHON ?= .venv/bin/python
EXPERIMENT ?= mscale_fno_100

.PHONY: docker-build docker-up doctor test pipeline status
docker-build:
	bash scripts/docker.sh compose build dolfinx-complex

docker-up:
	bash scripts/docker.sh compose up -d --wait --wait-timeout 180 dolfinx-complex

doctor:
	$(PYTHON) scripts/pipeline.py doctor --experiment $(EXPERIMENT)

test:
	$(PYTHON) scripts/pipeline.py test --experiment $(EXPERIMENT)

pipeline:
	$(PYTHON) scripts/pipeline.py run --experiment $(EXPERIMENT)

status:
	$(PYTHON) scripts/pipeline.py status --experiment $(EXPERIMENT)

.PHONY: sample-data sample-study sample-status
sample-data: docker-up
	bash scripts/docker.sh compose exec -T dolfinx-complex python3 -u -m experiments.multiscale_sample_study.run prepare

sample-study: sample-data
	OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $(SAMPLE_PYTHON) -u -m experiments.multiscale_sample_study.run all
	OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $(SAMPLE_PYTHON) -m experiments.multiscale_sample_study.audit
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -m experiments.multiscale_sample_study.label_audit
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -m experiments.multiscale_sample_study.baselines
	$(SAMPLE_PYTHON) -m experiments.multiscale_sample_study.run report

sample-status:
	$(SAMPLE_PYTHON) -m experiments.multiscale_sample_study.run status

.PHONY: highk-reference highk-data highk-train highk-baselines highk-verify highk-report highk-study highk-status
highk-reference: docker-up
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -u -m experiments.highk_svd_gmsfem.highorder --stage reference

highk-data: highk-reference
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -u -m experiments.highk_svd_gmsfem.run prepare

highk-train:
	VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $(SAMPLE_PYTHON) -u -m experiments.highk_svd_gmsfem.run train

.PHONY: highk-gms-refinement
highk-gms-refinement:
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -u -m experiments.highk_svd_gmsfem.gms_refinement --root experiments/highk_svd_gmsfem

highk-baselines:
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -u -m experiments.highk_svd_gmsfem.run baselines

highk-verify:
	VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $(SAMPLE_PYTHON) -m experiments.highk_svd_gmsfem.run verify
	bash scripts/docker.sh compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 dolfinx-complex python3 -u -m experiments.highk_svd_gmsfem.run audit-labels

highk-report:
	$(SAMPLE_PYTHON) -m experiments.highk_svd_gmsfem.run report

highk-study:
	$(MAKE) highk-data
	$(MAKE) highk-gms-refinement
	$(MAKE) highk-train
	$(MAKE) highk-baselines
	$(MAKE) highk-verify
	$(MAKE) highk-report

highk-status:
	$(SAMPLE_PYTHON) -m experiments.highk_svd_gmsfem.run status
