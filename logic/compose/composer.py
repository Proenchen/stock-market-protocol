from typing import List
from logic.analyzers.base import AnalyzerOutput


class Composer: 
    @staticmethod
    def compose_document(title: str, outputs: List[AnalyzerOutput]) -> str:
        """Compose a complete LaTeX document from analyzer outputs.

        Parameters
        ----------
        title : str
            The document title that will appear in the LaTeX header.
        outputs : list of AnalyzerOutput
            A list of outputs, each providing LaTeX blocks and a section name.

        Returns
        -------
        str
            The full LaTeX source as a single string, ready for compilation.
        """
        parts: list[str] = []
        for out in outputs:
            # Each output can contain multiple LaTeX blocks, all grouped under a section
            section_content = "\\section{" + out.name + "}\n" + "\n".join(out.latex_blocks)
            parts.append(section_content)

        # Document skeleton with header, packages, and title
        return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{float}
""" + "\\title{" + title + "}" + r"""
\date{\today}
\begin{document}
\maketitle
""" + "\n\n".join(parts) + "\n\\end{document}\n"
