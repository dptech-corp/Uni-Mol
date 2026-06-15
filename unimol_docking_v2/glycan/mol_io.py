"""Molecular I/O utilities for HMO and glycan ligands."""

from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem


def load_first_sdf_molecule(
    sdf_path: str,
    remove_hs: bool = False,
) -> Chem.Mol:
    """Load the first valid molecule from an SDF file."""

    path = Path(sdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"SDF file not found: {path}")

    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=remove_hs,
        sanitize=True,
    )

    for mol in supplier:
        if mol is not None:
            if mol.GetNumConformers() == 0:
                raise ValueError(
                    f"Molecule has no 3D conformer: {path}"
                )
            return mol

    raise ValueError(f"No valid molecule found in: {path}")


def prepare_heavy_atom_molecule(sdf_path: str) -> Chem.Mol:
    """
    Load an SDF molecule and standardize it to a heavy-atom representation.

    All downstream atom symbols, coordinates, linkage indices and torsion
    indices must be generated from this returned molecule.
    """

    mol = load_first_sdf_molecule(
        sdf_path=sdf_path,
        remove_hs=False,
    )

    # Preserve the original SDF atom index before hydrogen removal.
    for atom in mol.GetAtoms():
        atom.SetIntProp("_OriginalAtomIndex", atom.GetIdx())

    heavy_mol = Chem.RemoveHs(mol, sanitize=True)

    if heavy_mol.GetNumConformers() == 0:
        raise ValueError(
            f"Hydrogen removal lost conformer coordinates: {sdf_path}"
        )

    # Atom-map numbers are one-based and are only used for visualization.
    # All Python and JSON annotations remain zero-based.
    for atom in heavy_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    validate_molecule_coordinates(heavy_mol)

    return heavy_mol


def get_atom_symbols(mol: Chem.Mol) -> List[str]:
    """Return atom symbols in RDKit atom order."""

    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def get_coordinates(mol: Chem.Mol) -> np.ndarray:
    """Return coordinates with shape [num_atoms, 3]."""

    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    coordinates = np.asarray(
        mol.GetConformer().GetPositions(),
        dtype=np.float32,
    )

    validate_coordinate_array(
        coordinates=coordinates,
        num_atoms=mol.GetNumAtoms(),
    )

    return coordinates


def validate_coordinate_array(
    coordinates: np.ndarray,
    num_atoms: int,
) -> None:
    """Validate coordinate shape and numerical values."""

    if coordinates.shape != (num_atoms, 3):
        raise ValueError(
            "Coordinate shape mismatch: "
            f"expected {(num_atoms, 3)}, got {coordinates.shape}"
        )

    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates contain NaN or infinity")


def validate_molecule_coordinates(mol: Chem.Mol) -> None:
    """Validate molecule atom count and conformer coordinates."""

    coordinates = np.asarray(
        mol.GetConformer().GetPositions(),
        dtype=np.float32,
    )

    validate_coordinate_array(
        coordinates=coordinates,
        num_atoms=mol.GetNumAtoms(),
    )


def write_numbered_sdf(
    mol: Chem.Mol,
    output_path: str,
) -> None:
    """Write a molecule whose atom-map numbers show RDKit atom indices + 1."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(path))

    if writer is None:
        raise OSError(f"Could not create SDF writer: {path}")

    writer.write(mol)
    writer.close()
