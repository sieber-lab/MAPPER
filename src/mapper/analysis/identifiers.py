"""Traceable identifier matching to the original MAPPER reference genes.

Names, accessions and locus tags are matched through explicit FASTA, UniProt
and MG1655 GFF annotations. No sequence-similarity threshold or model outcome
is used to guess a gene identity. Ambiguous aliases remain ambiguous.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


def fasta_records(path):
    records, header, sequence = [], None, []
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence)))
                header, sequence = line[1:].strip(), []
            else:
                sequence.append(line.strip())
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def identifiers(text):
    if text is None or pd.isna(text):
        return []
    result = []
    for token in re.split(r"[;,\s]+", str(text).strip()):
        if not token or token in {"nan", "NA", "-"}:
            continue
        if token.startswith(("sp|", "tr|")):
            token = token.split("|")[1]
        result.append(token)
    return list(dict.fromkeys(result))


class Reference:
    def __init__(self, genes, fasta, alias_file, gff):
        self.genes = list(map(str, genes))
        self.targets = set(self.genes)
        self.names = defaultdict(set)
        self.accessions = defaultdict(set)
        self.entries = defaultdict(set)
        self.sequences = defaultdict(set)
        self.evidence = defaultdict(set)
        for gene in self.genes:
            self.names[gene].add(gene)
        seen = set()
        for header, sequence in fasta_records(fasta):
            match = re.search(r"\bGN=([^\s]+)", header)
            if not match or match[1] not in self.targets:
                continue
            gene = match[1]
            fields = header.split()[0].split("|")
            accession = fields[1] if len(fields) >= 2 else fields[0]
            entry = fields[2] if len(fields) >= 3 else ""
            self.accessions[accession].add(gene)
            if entry:
                self.entries[entry].add(gene)
                self.entries[entry.split("_")[0]].add(gene)
            self.sequences[hashlib.sha256(sequence.encode()).hexdigest()].add(gene)
            self.evidence[gene].add("training FASTA:" + accession)
            seen.add(gene)
        if seen != self.targets:
            raise ValueError("Reference FASTA does not cover the exact training universe")
        alias = pd.read_csv(alias_file, sep="\t").fillna("")
        for _, row in alias.iterrows():
            names = identifiers(row["Gene Names (primary)"]) + identifiers(
                row["Gene Names (synonym)"]
            )
            # Historical aliases can name different modern genes (e.g. csdA).
            # The reference protein accession anchors identity; do not merge
            # two proteins merely because they once shared a name.
            candidates = set(self.accessions.get(row.Entry, ()))
            if not candidates and row["Gene Names (primary)"] in self.targets:
                candidates = {row["Gene Names (primary)"]}
            if not candidates:
                candidates = set(names) & self.targets
            if not candidates:
                continue
            self.accessions[row.Entry].update(candidates)
            self.entries[row["Entry Name"]].update(candidates)
            self.entries[row["Entry Name"].split("_")[0]].update(candidates)
            for name in names:
                self.names[name].update(candidates)
            for gene in candidates:
                self.evidence[gene].add("UniProt:" + row.Entry)
        gene_info, cds_info = {}, []
        with gzip.open(gff, "rt") as stream:
            for line in stream:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9 or fields[2] not in {"gene", "CDS"}:
                    continue
                attrs = {
                    k: unquote(v)
                    for item in fields[8].split(";")
                    if "=" in item
                    for k, v in [item.split("=", 1)]
                }
                if fields[2] == "gene":
                    gene_info[attrs["ID"]] = attrs
                else:
                    cds_info.append(attrs)
        self.loci = {}
        for attrs in cds_info:
            locus = attrs.get("Parent", "")
            info = gene_info.get(locus, {})
            names = {
                info.get("gene", ""),
                *info.get("gene_synonym", "").split(","),
                info.get("locus_tag", ""),
            }
            refs = re.findall(r"(?:Swiss-Prot|TrEMBL):([^,;]+)", attrs.get("Dbxref", ""))
            candidates = set()
            for accession in refs:
                candidates.update(self.accessions.get(accession, ()))
            if not candidates:
                primary = info.get("gene", "")
                candidates = set(self.names.get(primary, ()))
            if not candidates:
                candidates = set(names) & self.targets
            if not candidates:
                continue
            self.loci[locus] = candidates
            for name in names - {""}:
                self.names[name].update(candidates)
            for accession in [attrs.get("protein_id", ""), *refs]:
                if accession:
                    self.accessions[accession].update(candidates)
            for gene in candidates:
                self.evidence[gene].add("MG1655 GFF:" + locus)

    def resolve(self, gene_text="", accession_text="", entry_text="", allow_multiple=False):
        tokens = identifiers(gene_text)
        exact = set(tokens) & self.targets
        protein = set().union(
            *(self.accessions.get(token, set()) for token in identifiers(accession_text))
        )
        entries = set()
        for token in identifiers(entry_text):
            entries.update(self.entries.get(token, ()))
            entries.update(self.entries.get(token.split("_")[0], ()))
        named = set()
        anchor = protein or entries
        for token in tokens:
            candidates = {token} if token in self.targets else set(self.names.get(token, ()))
            # A shared historical name is resolved by the actual protein ID.
            # E.g. ribE;ribH with RISB accessions must not also label ribC.
            if len(candidates) > 1 and anchor and candidates & anchor:
                candidates &= anchor
            named.update(candidates)
        if allow_multiple and len(tokens) > 1 and named:
            choice, method = named, (
                "explicit_shared_gene_group" if len(named) > 1 else "unique_annotation_alias"
            )
        elif exact:
            choice, method = exact, "exact_gene"
        elif len(named) == 1:
            choice, method = named, "unique_annotation_alias"
        elif len(protein) == 1 and (not named or protein <= named):
            choice, method = protein, "unique_reference_accession"
        elif (
            len(entries) == 1
            and (not named or entries <= named)
            and (not protein or entries <= protein)
        ):
            choice, method = entries, "unique_uniprot_entry"
        elif allow_multiple and named:
            choice, method = named, "explicit_shared_gene_group"
        else:
            choice, method = set(), (
                "ambiguous" if named or protein or entries else "unresolved_identifier"
            )
        conflict = bool(choice and protein and choice.isdisjoint(protein))
        if conflict:
            choice, method = set(), "gene_accession_conflict"
        if len(choice) > 1 and not allow_multiple:
            choice, method = set(), "shared_protein_group"
        return dict(
            targets=sorted(choice),
            rule=method,
            name_candidates=";".join(sorted(named)),
            accession_candidates=";".join(sorted(protein)),
            entry_candidates=";".join(sorted(entries)),
        )

    def table(self):
        rows = []
        for gene in self.genes:
            rows.append(
                dict(
                    gene=gene,
                    unambiguous_names=";".join(
                        sorted(k for k, v in self.names.items() if v == {gene})
                    ),
                    unambiguous_accessions=";".join(
                        sorted(k for k, v in self.accessions.items() if v == {gene})
                    ),
                    annotation_evidence=";".join(sorted(self.evidence[gene])),
                )
            )
        return pd.DataFrame(rows)

    def add_source_records(self, table):
        """Use exact sequences, or unique explicit names in source annotations."""
        audit = []
        for _, row in table.fillna("").iterrows():
            candidates = (
                self.sequences.get(hashlib.sha256(row.sequence.encode()).hexdigest(), set())
                if row.sequence
                else set()
            )
            method = "identical_reference_sequence" if candidates else ""
            if not candidates:
                source_names = identifiers(row.source_genes)
                named = set().union(*(self.names.get(name, set()) for name in source_names))
                if len(named) == 1:
                    candidates, method = named, "unique_source_record_gene"
            if candidates:
                self.accessions[row.accession] = set(candidates)
                # Numeric UniProt-style entry names in the old gentamicin
                # workbook are anchored by the accession, not their label.
                self.entries[row.accession] = set(candidates)
                for gene in candidates:
                    self.evidence[gene].add(row.authority + ":" + row.accession)
            audit.append(
                dict(
                    accession=row.accession,
                    organism=row.organism,
                    source_genes=row.source_genes,
                    reference_candidates=";".join(sorted(candidates)),
                    rule=method or "no_unique_reference_identity",
                    source_file=row.record_file,
                )
            )
        return pd.DataFrame(audit)


def map_table(
    reference,
    table,
    dataset,
    gene_column,
    accession_column=None,
    entry_column=None,
    expand=False,
    excluded=None,
):
    rows = []
    for index, source in table.reset_index(drop=True).iterrows():
        gene = source.get(gene_column, "") if gene_column else ""
        accession = source.get(accession_column, "") if accession_column else ""
        entry = source.get(entry_column, "") if entry_column else ""
        resolved = reference.resolve(gene, accession, entry, allow_multiple=expand)
        if excluded is not None and bool(excluded[index]):
            resolved["targets"], resolved["rule"] = [], "source_identification_qc_exclusion"
        rows.append(
            dict(
                dataset=dataset,
                source_row=index,
                source_gene="" if pd.isna(gene) else str(gene),
                source_accession="" if pd.isna(accession) else str(accession),
                source_entry="" if pd.isna(entry) else str(entry),
                mapped_genes=";".join(resolved.pop("targets")),
                **resolved,
            )
        )
    return pd.DataFrame(rows)
