"""Воспроизводимая геометрия Gmsh с 24 заполненными круглыми включениями.

Единственный источник CAD — ``geo/scattering.geo``. Python задаёт параметры,
присваивает материал уже созданным OCC-граням и сохраняет сетку. Разбиение
квадрата на крупные ячейки входит в BooleanFragments, поэтому рёбра мелкой
сетки согласованы с границами крупных ячеек.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

_GEO_DIR = Path(__file__).resolve().parent / "geo"


def inclusion_circles() -> np.ndarray:
    """Вернуть независимый массив ``(24, 4)``: x, y, радиус, physical tag.

    Теги 2/3/4 обозначают радиусы 0.02/0.05/0.10 соответственно; фон имеет
    тег 1. Координаты точно следуют исходному ``данные/сетки/mesh.geo``.
    """
    # Те же массивы, которые читает Gmsh: второго списка центров в Python нет.
    source = (_GEO_DIR / "scattering.geo").read_text()
    arrays = []
    for name in ("cx", "cy", "ss"):
        match = re.search(rf"\b{name}\[\]\s*=\s*\{{([^}}]+)\}}\s*;", source, re.S)
        if match is None:
            raise ValueError(f"В scattering.geo отсутствует массив {name}[]")
        arrays.append(np.array([float(value.strip()) for value in match.group(1).split(",")]))
    if not (len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == 24):
        raise ValueError("В scattering.geo ожидаются 24 включения в массивах cx, cy, ss")
    raw = np.column_stack(arrays)
    return np.column_stack((raw[:, 0] * .004 - 2, 2 - raw[:, 1] * .004,
                            raw[:, 2] * .002,
                            np.select([raw[:, 2] == 10, raw[:, 2] == 25],
                                      [2, 3], default=4)))


def _interior_point(gmsh, surface: int) -> np.ndarray:
    """Найти точку внутри обрезанной OCC-грани, а не внутри её bbox.

    Центр масс кольцевой фоновой грани может попасть в отверстие. Поэтому
    он обязательно проверяется ``isInside``; запасной поиск использует
    параметризацию плоскости и ту же проверку обрезанной границы.
    """
    point = np.asarray(gmsh.model.occ.getCenterOfMass(2, surface))
    if gmsh.model.isInside(2, surface, point.tolist()):
        return point
    lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, surface))
    for count in (5, 13, 37, 101):
        # Несимметричные точки исключают попадание на интерфейс/вершину.
        for i in range(count):
            for j in range(count):
                uv = lo + (hi-lo) * np.array(((i+.371)/count, (j+.619)/count))
                point = np.asarray(gmsh.model.getValue(2, surface, uv.tolist()))
                if gmsh.model.isInside(2, surface, point.tolist()):
                    return point
    raise RuntimeError(f"Не найдена внутренняя точка OCC-грани {surface}")


def _assign_materials(gmsh) -> dict[str, float]:
    """Присвоить граням материал по настоящей внутренней точке и проверить площади."""
    circles = inclusion_circles()
    groups = {tag: [] for tag in (1, 2, 3, 4)}
    areas = {tag: 0. for tag in groups}
    for _, surface in gmsh.model.getEntities(2):
        point = _interior_point(gmsh, surface)
        inside = np.sum((circles[:, :2] - point[:2])**2, axis=1) < circles[:, 2]**2
        matches = circles[inside, 3]
        tag = int(matches[0]) if len(matches) else 1
        groups[tag].append(surface)
        areas[tag] += gmsh.model.occ.getMass(2, surface)
    for tag, name in ((1, "background"), (2, "small"), (3, "medium"), (4, "large")):
        gmsh.model.addPhysicalGroup(2, groups[tag], tag)
        gmsh.model.setPhysicalName(2, tag, name)
        if tag > 1:
            expected = float(np.sum(np.pi * circles[circles[:, 3] == tag, 2]**2))
            if not np.isclose(areas[tag], expected, rtol=1e-7, atol=1e-10):
                raise RuntimeError(f"Неверная площадь материала {tag}: {areas[tag]} != {expected}")
    if not np.isclose(sum(areas.values()), 16., atol=1e-9):
        raise RuntimeError("Суммарная площадь CAD не равна 16")
    return {str(k): v for k, v in areas.items()}


def generate_mesh(output_path: Path, *, n_coarse: int = 10, h: float = .04,
                  reference: bool = False) -> Path:
    """Создать или прочитать из кэша согласованную треугольную сетку MSH 4.1 ASCII.

    ``output_path`` — имя .msh; ``n_coarse`` задаёт число квадратов вдоль оси,
    ``h`` — размер элементов. При ``reference=True`` эффективный размер h/2,
    геометрия и маркировка совпадают с основной сеткой. Параметры передаются
    интерпретатору GEO через ONELAB/DefineConstant до ``gmsh.open``;
    обычный parser.setNumber не подходит, поскольку open очищает parser.

    Рядом сохраняется ``<имя>.mesh_info.json`` с параметрами, SHA256 исходных
    GEO и готового MSH, площадями CAD и версией Gmsh. Кэш принимается только
    при совпадении всех параметров и контрольных сумм. Gmsh должен вызываться
    на одном процессе (обычно MPI rank 0), после чего сетку можно читать MPI.
    """
    if isinstance(n_coarse, bool) or int(n_coarse) != n_coarse or n_coarse < 1:
        raise ValueError("n_coarse должен быть положительным целым")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h должен быть конечным положительным числом")
    import gmsh

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = output_path.with_suffix(".mesh_info.json")
    digest = hashlib.sha256()
    for source in sorted(_GEO_DIR.glob("*.geo")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    signature = dict(n_coarse=int(n_coarse), h=float(h), reference=bool(reference),
                     effective_h=float(h / 2 if reference else h),
                     geo_sha256=digest.hexdigest(),
                     generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     gmsh_version=gmsh.__version__)
    if output_path.exists() and info_path.exists():
        try:
            old = json.loads(info_path.read_text())
            if all(old.get(k) == v for k, v in signature.items()) and old.get("msh_sha256") == hashlib.sha256(output_path.read_bytes()).hexdigest():
                return output_path
        except (ValueError, OSError):
            pass
    own_session = not gmsh.isInitialized()
    if own_session:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    verbosity = gmsh.option.getNumber("General.Verbosity")
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("gmsfem_geometry")
        gmsh.onelab.setNumber("GMSFEM/n_coarse", [int(n_coarse)])
        gmsh.onelab.setNumber("GMSFEM/h", [signature["effective_h"]])
        gmsh.open(str(_GEO_DIR / ("reference.geo" if reference else "scattering.geo")))
        if (int(gmsh.parser.getNumber("n_coarse")[0]) != n_coarse
                or float(gmsh.parser.getNumber("h")[0]) != signature["effective_h"]):
            raise RuntimeError("Gmsh не применил параметры GEO")
        areas = _assign_materials(gmsh)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(output_path))
        info = dict(signature, material_areas=areas,
                    msh_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest())
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    finally:
        if own_session:
            gmsh.finalize()
        else:
            gmsh.model.remove()
            gmsh.option.setNumber("General.Verbosity", verbosity)
            if previous_model:
                gmsh.model.setCurrent(previous_model)
    return output_path
