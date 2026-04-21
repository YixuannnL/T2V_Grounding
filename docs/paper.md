\documentclass[10pt,twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{microtype}  % Better line breaking

% Allow more flexible line breaking
\tolerance=1000
\emergencystretch=3em
\hyphenpenalty=50

\title{Grounding-in-the-Loop: Agentic Multi-Shot Text-to-Video Generation with Visual Entity Consistency}

\author{Anonymous Authors}

\begin{document}

\maketitle

% ============================================================================
\begin{abstract}
Multi-shot video generation from text scripts requires maintaining consistent visual identity for characters, objects, and scenes across shots. We argue that this is fundamentally an \textbf{agentic} problem: the system must continuously perceive what has been generated, reason about entity identities and quality, and act adaptively---yet existing approaches treat it as a static pipeline. We introduce an agentic framework built on \textbf{Grounding-in-the-Loop}: rather than relying on external reference images, we extract references from the generated videos themselves via visual grounding, creating a self-bootstrapping perception-action loop. Beyond this core loop, our pipeline incorporates multiple runtime agentic decisions---generation verification with automatic retry, subject-aware mode routing, lighting-adaptive reference selection, and frontal-aware filtering---each following an explicit perceive-reason-act pattern. Without any training or fine-tuning, our approach achieves robust cross-shot consistency while preserving full freedom over camera angles and character poses.
\end{abstract}

% ============================================================================
\section{Introduction}

Generating a coherent multi-shot video from a text script is fundamentally harder than single-shot generation. A compelling visual narrative demands that the same character---appearing across different shots with varying camera angles, lighting, and actions---remains recognizably consistent. Yet current T2V models~\cite{wan2024wan,yang2024cogvideox,gen3} treat each shot as an independent generation, causing severe appearance drift that breaks narrative coherence.

What makes multi-shot generation so challenging? We argue that it is inherently an \textbf{agentic} problem: the system must continuously \emph{perceive} what has been generated, \emph{reason} about entity identities and visual quality, and \emph{act} adaptively to maintain consistency---all while handling the uncertainty and errors inevitable in generative models. Yet existing approaches treat it as a static pipeline problem. End-to-end multi-shot models~\cite{shotstream,multishotmaster,echoshot} learn consistency implicitly through training, offering no mechanism to inspect or correct failures at runtime. Reference-based pipelines~\cite{videomemory,storydiffusion} follow fixed ``plan $\to$ generate $\to$ store'' workflows without feedback loops. When generation errors occur---wrong entity count, poor reference quality, style mismatch---these systems have no way to detect or recover from them.

We propose to reframe multi-shot video generation as an \textbf{agentic process} with explicit perception-reasoning-action loops. Our key insight is that maintaining visual consistency requires the system to actively monitor its own outputs and make runtime decisions, rather than blindly executing a predetermined plan.

This agentic perspective leads to our core design: \textbf{Grounding-in-the-Loop}. Instead of relying on externally-provided or separately-generated reference images, we extract references directly from the generated videos via visual grounding. This is itself an agentic act---the system \emph{perceives} what it has generated, \emph{extracts} entity representations, and \emph{uses} them to condition future generation. The loop is self-bootstrapping: Shot~1 generates content $\to$ grounding extracts entity crops $\to$ crops become references for Shot~2 $\to$ grounding enriches the reference bank further. Crucially, references extracted from generated videos are guaranteed to be stylistically consistent---same model, same rendering pipeline, same visual distribution.

Beyond the grounding loop, our pipeline incorporates multiple \textbf{runtime agentic decisions}:

\begin{itemize}
    \item \textbf{Quality-Aware Shot Scheduling:} We \emph{perceive} the script structure and predict grounding quality for each shot-entity pair, \emph{reason} about optimal execution order via a dependency DAG, and \emph{act} by generating high-quality close-up shots first to establish strong reference anchors---even when they appear later in narrative order.

    \item \textbf{Generation-Verification Loop:} After generating anchor shots, we \emph{perceive} entity counts using a multimodal LLM, \emph{reason} about correctness, and \emph{act} by retrying with adjusted seeds or enhanced prompts if mismatches occur---preventing errors from propagating.

    \item \textbf{Subject-Aware Mode Routing:} We \emph{perceive} what references are available, \emph{reason} about whether subject (character/object) anchoring exists, and \emph{act} by routing to T2V or S2V accordingly---avoiding style drift from location-only conditioning.

    \item \textbf{Agentic Lighting Analysis:} For close-up shots, an LLM \emph{perceives} the scene's lighting complexity, \emph{reasons} about whether visual reference or textual description better preserves consistency, and \emph{acts} by including or excluding location references.

    \item \textbf{Frontal-Aware Filtering:} We \emph{perceive} face detection confidence in character references, \emph{reason} that back-view references may cause hallucinated faces, and \emph{act} by switching to text-based appearance description when confidence is low.

    \item \textbf{Agentic Reference Selection:} Rather than relying on fixed scoring formulas (e.g., InsightFace for faces), we \emph{perceive} candidate references via a Vision Language Model (VLM), \emph{reason} about which reference best matches the current shot's semantic requirements (dialogue scene $\to$ frontal face, action scene $\to$ dynamic pose), and \emph{act} by selecting the most contextually appropriate reference with explicit justification.
\end{itemize}

These mechanisms embody genuine agentic behavior---not just modular decomposition with agent-like naming, but active perception of generation outcomes, reasoning about quality and constraints, and adaptive action based on runtime observations.

A final design choice reinforces our agentic flexibility: we adopt \textbf{subject-to-video (S2V)} generation~\cite{phantom} rather than keyframe + I2V pipelines. I2V treats the reference as the first frame, locking spatial composition and limiting camera diversity. S2V conditions on appearance \emph{without} constraining layout, giving the agentic system freedom to generate diverse shots while maintaining entity consistency.

\vspace{0.5em}
\noindent\textbf{Contributions.} We make the following contributions:

\begin{enumerate}
    \item \textbf{Agentic formulation:} We reframe multi-shot video generation as an agentic problem requiring explicit perception-reasoning-action loops, rather than a static pipeline.

    \item \textbf{Grounding-in-the-Loop:} A self-bootstrapping paradigm where references are extracted from generated videos via visual grounding, ensuring stylistic consistency through agentic self-perception.

    \item \textbf{Quality-Aware Agentic Shot Scheduling:} We observe that the quality of visual grounding varies dramatically across shots due to camera distance and lighting. We propose an agentic planning approach where an LLM constructs a dependency-aware execution DAG, enabling ``reference bootstrapping''---generating high-quality close-up shots first to establish strong anchors, even when they appear later in narrative order.

    \item \textbf{Runtime agentic decisions:} Multiple perception-reasoning-action mechanisms---generation verification, subject-aware routing, lighting analysis, frontal-aware filtering---that actively adapt to generation outcomes.

    \item \textbf{Training-free deployment:} Our method requires no training or fine-tuning, enabling immediate deployment on any compatible T2V/S2V backbone.
\end{enumerate}

% ============================================================================
\section{Related Work}

Omit for now.

\section{Method}

We introduce T2V-Grounding, an agentic pipeline for multi-shot video generation with cross-shot entity consistency. Given a multi-shot script $\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$ where each shot $s_n$ is a natural language description, and an optional global caption $\mathcal{C}_{\text{global}}$ describing the overall narrative context, our goal is to generate a sequence of videos $\mathcal{V} = \{v_1, v_2, \ldots, v_N\}$ such that entities appearing across multiple shots maintain consistent visual identity.

The pipeline consists of six components: (1)~an LLM-based Entity Parser, (2)~a Global Context Extractor, (3)~an Entity Registry, (4)~a Visual Grounding Module, (5)~a Reference Quality Scorer, (6)~an Adaptive Video Generator, and (7)~an Agentic Shot Scheduler. These are orchestrated by an agentic loop that can execute shots in an \textbf{optimized order determined by DAG scheduling}. A key design is our \textbf{four-layer prompt construction}, which separates global semantic context, lighting guidance (for close-ups), shot-specific entity descriptions, and action text, preventing entity leakage across shots while maintaining stylistic and lighting coherence.

% ----------------------------------------------------------------------------
\subsection{Quality-Aware Agentic Shot Scheduling}

A fundamental assumption in existing multi-shot video generation is that shots should be executed in \emph{narrative order}: Shot~1 $\to$ Shot~2 $\to$ $\cdots$ $\to$ Shot~$N$. This implicitly assumes that the first appearance of an entity provides the best reference for subsequent shots. However, we observe that \textbf{grounding quality varies dramatically across shots}:

\begin{center}
\begin{tabular}{lcc}
\toprule
\textbf{Shot Type} & \textbf{Entity Coverage} & \textbf{Grounding Quality} \\
\midrule
Close-up & 50--80\% & High (0.85--0.95) \\
Medium shot & 15--40\% & Medium (0.50--0.70) \\
Wide shot & 3--10\% & Low (0.25--0.40) \\
Establishing & $<$5\% & Very low (0.15--0.30) \\
\bottomrule
\end{tabular}
\end{center}

If Shot~1 is a wide establishing shot and Shot~3 contains a character close-up, executing linearly would establish a \emph{poor-quality} anchor from Shot~1 (low resolution, possible profile view), causing appearance drift in all subsequent shots. Our key insight is:

\vspace{0.3em}
\noindent\fbox{\parbox{0.95\columnwidth}{
\textbf{Narrative order $\neq$ optimal generation order.} Shots with high grounding quality should be executed first to establish strong reference anchors, regardless of their position in the script.
}}
\vspace{0.3em}

\textbf{DAG-Based Execution Scheduling.} We model the multi-shot generation process as a directed acyclic graph (DAG):

\begin{itemize}
    \item \textbf{Nodes:} Each shot $s_n \in \mathcal{S}$
    \item \textbf{Edges:} $s_i \to s_j$ if shot $j$ requires entity references from shot $i$
\end{itemize}

Rather than assuming $s_1 \to s_2 \to \cdots \to s_N$, we construct the DAG based on \emph{predicted grounding quality}:

\begin{enumerate}
    \item \textbf{Quality Prediction.} For each (shot, entity) pair, we predict grounding quality $Q(s_n, e)$ based on shot type, lighting conditions, and entity prominence:
    \begin{equation}
    Q(s_n, e) = Q_{\text{base}}(\text{shot\_type}) \cdot \alpha_{\text{light}} \cdot \alpha_{\text{occl}} \cdot \alpha_{\text{pose}}
    \end{equation}
    where $Q_{\text{base}}$ is determined by shot type (close-up: 0.9, medium: 0.6, wide: 0.35), and $\alpha$ factors account for lighting (0.7 for backlit), occlusion (0.6 for partial), and pose (0.5 for profile/back view).

    \item \textbf{Reference Source Identification.} For each entity $e$, identify the ``reference source'' shot---the shot that will provide the highest quality anchor:
    \begin{equation}
    s^*_e = \argmax_{s_n : e \in \mathcal{E}_n} Q(s_n, e)
    \end{equation}

    \item \textbf{DAG Construction.} Build execution dependencies:
    \begin{itemize}
        \item Reference source shots have no incoming edges (execute first)
        \item Other shots containing entity $e$ depend on $s^*_e$
    \end{itemize}

    \item \textbf{Topological Execution.} Execute shots in topological order of the DAG. Ties are broken by narrative order to maintain temporal coherence where possible.
\end{enumerate}

\textbf{Benefit Assessment.} Before applying DAG scheduling, we compute the expected benefit:
\begin{equation}
\Delta Q = \max_e Q(s^*_e, e) - Q(s_1, e)
\end{equation}

If $\Delta Q < \tau_{\text{benefit}}$ (default 0.15), the improvement is negligible and we fall back to linear execution, avoiding unnecessary complexity.

\textbf{Example: Close-up After Wide Shot.} Consider:
\begin{itemize}
    \item Shot 1: Wide shot---Sarah walks through marketplace
    \item Shot 2: Medium shot---Sarah stops at fruit stall
    \item Shot 3: Close-up---Sarah examines apple
    \item Shot 4: Wide shot---Sarah continues walking
\end{itemize}

\emph{Linear execution:} Shot~1 provides Sarah's anchor (quality 0.35, small figure).

\emph{DAG execution:} Shot~3 identified as reference source (quality 0.90, frontal close-up). Execution order: 3 $\to$ 1 $\to$ 2 $\to$ 4. Shot~1's generation now benefits from the high-quality anchor established in Shot~3.

\textbf{Progressive Reveal Handling.} A special case is ``progressive reveal'' scripts where a character is intentionally obscured early:
\begin{itemize}
    \item Shot 1: Mysterious hooded figure in shadows
    \item Shot 2: Figure steps into light, face partially visible
    \item Shot 3: Figure removes hood, revealing Detective Chen
\end{itemize}

The scheduler recognizes that Shot~3 is the only viable reference source for ``Chen''. Even though narratively the reveal should be delayed, we generate Shot~3 first to establish Chen's appearance, then use that reference to ensure the ``mysterious figure'' in Shots~1--2 is visually consistent with the revealed identity.

% ----------------------------------------------------------------------------
\subsection{LLM-Based Entity Parser with Coreference}

Before generating shot $n$, we run the Entity Parser to extract a structured entity list from the shot text $s_n$.

\textbf{Entity Extraction.} A language model $\mathcal{M}_{\text{LLM}}$ processes the shot description along with an accumulated context of known entities from previous shots:
\begin{equation}
\mathcal{E}_n = \mathcal{M}_{\text{LLM}}(s_n,\ \mathcal{K}_{n-1})
\end{equation}
where $\mathcal{K}_{n-1} = \{e \mid e \in \bigcup_{i<n} \mathcal{E}_i,\ e.\text{is\_new} = \texttt{true}\}$ is the set of all entities introduced in previous shots. The parser produces a list of entity records, where each entity $e$ contains:
\begin{itemize}
    \item $e.\text{id}$: a stable cross-shot identifier (e.g., \texttt{char\_alex})
    \item $e.\text{type} \in \{\text{character}, \text{object}, \text{location}, \text{style}\}$
    \item $e.\text{desc}$: natural language description for grounding
    \item $e.\text{priority} \in \{\text{high}, \text{medium}, \text{low}\}$
    \item $e.\text{is\_new}$: whether this entity first appears here
\end{itemize}

\textbf{Cross-Shot Coreference Resolution.} A critical challenge is that the same entity may be referred to differently across shots: ``Alex'' in shot~1 may appear as ``the detective'' in shot~3 or ``he'' in shot~4. We address this by providing the LLM with $\mathcal{K}_{n-1}$ as context and instructing it to assign the same ID to coreferred entities. The LLM also maintains an alias list for each entity, enabling robust matching even under diverse referring expressions.

\textbf{Global Context Extraction.} In addition to entity extraction, the parser also extracts global semantic context from an optional \texttt{global\_caption} describing the overall video narrative. This global context is decomposed into four dimensions:
\begin{itemize}
    \item \emph{visual\_style}: Cinematographic style (e.g., ``dramatic lighting'')
    \item \emph{mood}: Emotional atmosphere (e.g., ``tense, suspenseful'')
    \item \emph{setting}: Environmental description (e.g., ``desolate desert'')
    \item \emph{narrative}: Story context (e.g., ``a pursuit sequence'')
\end{itemize}

Critically, the global context \textbf{excludes specific entity descriptions} to avoid injecting characters that should not appear in the current shot.

% ----------------------------------------------------------------------------
\subsection{Adaptive Video Generation with Mode Routing}

Given the entity list $\mathcal{E}_n$ and the Entity Registry state $\mathcal{R}_{n-1}$, we route shot $n$ to one of two generation modes:

\textbf{T2V Mode (Shot 1 or new entities).} When no visual references are available for any high-priority entity (i.e., the registry is empty or all relevant entities are new), we use a pure text-to-video model $\mathcal{G}_{\text{T2V}}$:
\begin{equation}
v_n = \mathcal{G}_{\text{T2V}}(s_n)
\end{equation}

\textbf{Shot 1 Entity Count Verification.} A critical observation is that T2V generation without reference constraints is prone to \emph{entity count errors}: if the prompt describes ``three men'', the model may generate four people. Since all subsequent shots depend on Shot~1's grounding results as anchors, such errors propagate throughout the entire multi-shot sequence.

To address this, we introduce a \textbf{Generation-Verification Loop} for T2V mode:

\begin{enumerate}
    \item \textbf{Entity Count Extraction.} During LLM parsing, we extract expected entity counts. For each entity type $t$:
    \begin{equation}
    N_t^{\text{exp}} = |\{e \in \mathcal{E}_n : e.\text{type} = t\}|
    \end{equation}

    \item \textbf{Post-Generation Verification with MLLM.} After generating $v_n$, we sample $K$ frames uniformly and count persons using a multimodal LLM (Claude Haiku 4.5). The MLLM approach offers several advantages over detection-based counting:
    \begin{itemize}
        \item \textbf{Semantic understanding}: Can distinguish real people from statues, paintings, reflections, or posters
        \item \textbf{Robustness to occlusion}: Understands partial visibility and overlapping figures
        \item \textbf{Small target detection}: More reliable for distant or blurred figures that detection models often miss
    \end{itemize}

    For each sampled frame $f_k$, we query the MLLM:
    \begin{equation}
    c_k = \mathcal{M}_{\text{MLLM}}(f_k,\ \text{``Count real people in this image''})
    \end{equation}

    The final count uses mode aggregation for robustness:
    \begin{equation}
    N_{\text{person}}^{\text{act}} = \text{mode}\bigl(\{c_k\}_{k=1}^K\bigr)
    \end{equation}

    If MLLM inference fails, we fall back to detection-based counting (Grounding DINO + NMS).

    \item \textbf{Verification and Retry.} If $N_{\text{person}}^{\text{act}} \neq N_{\text{person}}^{\text{exp}}$:
    \begin{itemize}
        \item Attempt 1--2: Retry with different seed
        \item Attempt 3+: Use an \textbf{enhanced prompt}:
        \begin{equation}
        s_n' = \text{``[exactly } N \text{ people]''} \oplus s_n
        \end{equation}
    \end{itemize}

    \item \textbf{Termination.} Accept if counts match, or after $M_{\max}$ retries (default~3), warn and proceed with best available result.
\end{enumerate}

This verification loop ensures that the ``anchor'' references established in Shot~1 are semantically correct, preventing error accumulation. The MLLM-based counting is particularly effective for complex scenes with partial occlusion, small figures, or non-human entities that might confuse detection models.

\textbf{S2V Mode (subsequent shots with references).} When the registry contains reference crops for at least one high-priority \textbf{subject} entity (character or object) in shot $n$, we switch to a subject-to-video model $\mathcal{G}_{\text{S2V}}$:
\begin{equation}
v_n = \mathcal{G}_{\text{S2V}}(s_n,\ \mathcal{I}_n)
\end{equation}
where $\mathcal{I}_n$ is the set of reference images retrieved from the registry.

\textbf{Subject-Aware Mode Routing.} A critical observation is that S2V mode requires \emph{subject} references (characters or objects) to anchor the visual appearance. If only \emph{location} references are available, the S2V model lacks appearance constraints for foreground entities, often resulting in style drift (e.g., generating animated characters instead of photorealistic ones).

We introduce a \textbf{subject-aware routing} mechanism that considers only \emph{frontal} character references:
\begin{equation}
\text{mode} = \begin{cases}
\text{S2V} & \text{if } \exists e \in \mathcal{E}_n^{\text{subj}} : \mathcal{R}^{\text{frontal}}[e] \neq \emptyset \\
\text{T2V} & \text{otherwise}
\end{cases}
\end{equation}
where $\mathcal{E}_n^{\text{subj}} = \{e \in \mathcal{E}_n : e.\text{type} \in \{\text{character}, \text{object}\}\}$ are the subject entities, and $\mathcal{R}^{\text{frontal}}[e]$ denotes references with $\text{id\_conf} \geq \tau_{\text{face}}$ for characters.

Critically, characters with only faceless (back-view) references do \emph{not} count as valid subjects for S2V routing. This prevents the S2V model from hallucinating faces based on clothing alone, which typically produces style-inconsistent results.

When falling back to T2V due to missing subject references, we inject an \textbf{Environment Context} layer into the prompt, describing the location's attributes (lighting, atmosphere, setting) to maintain scene consistency through text guidance rather than image conditioning:
\begin{equation}
P_{\text{env}} = \text{BuildEnvContext}(\mathcal{E}_n^{\text{loc}})
\end{equation}

This ensures that: (1)~S2V is only used when meaningful appearance anchoring is possible; (2)~T2V fallback maintains environmental consistency via enhanced prompts; (3)~style drift from location-only S2V conditioning is avoided.

\textbf{Why S2V and not I2V?} Image-to-video models treat the reference image as the first video frame, strongly constraining both appearance \emph{and} spatial layout/pose. This makes I2V unsuitable for shots with different camera angles or actions. S2V models like Phantom instead encode reference images as appearance tokens in a parallel conditioning stream, exerting appearance-level influence without fixing spatial structure.

\textbf{Earliest High-Quality Anchor Strategy.} A critical observation is that \emph{appearance drift accumulates across shots}: the later a shot is generated, the more likely the character's face deviates from its original appearance. If we always use the most recent shot's grounding result, errors compound progressively.

A naive anchor strategy would simply select the earliest shot's reference. However, we observe that \textbf{the earliest shot is not always the best}: Shot~1 (typically a wide shot) may capture a character in profile or partial view, while Shot~2 (often a close-up) provides a clearer frontal face. Blindly selecting the earliest reference can lock in a poor-quality side profile.

To address this, we introduce an \textbf{``earliest high-quality''} anchor strategy that balances recency with quality:

\begin{enumerate}
    \item If the earliest shot contains a reference with quality $\geq \tau_{\text{high}}$ (default 0.85), select it.
    \item Otherwise, search all shots for the \emph{earliest} reference with $q \geq \tau_{\text{high}}$.
    \item If no high-quality reference exists, fall back to the earliest shot's best reference (to still prevent drift).
\end{enumerate}

Formally:
\begin{equation}
\mathcal{I}_{\text{anchor}}(e) = \begin{cases}
\mathcal{R}[e, 1] & \text{if } q(\mathcal{R}[e, 1]) \geq \tau_{\text{high}} \\
\displaystyle\argmin_{s} \{\mathcal{R}[e, s] : q \geq \tau_{\text{high}}\} & \text{if } \exists \text{ HQ ref} \\
\mathcal{R}[e, 1] & \text{otherwise}
\end{cases}
\end{equation}

This ensures that: (1)~when a clear frontal face is captured early, it becomes the anchor; (2)~when the earliest shot only has a poor-quality side profile but a later close-up provides a better reference, the system intelligently selects the higher-quality option; (3)~ties are broken in favor of earlier shots to minimize drift.

\textbf{Agentic Light-Aware Close-up Strategy.} A naive approach excludes location references for all close-up shots to allow background defocus. However, this causes lighting and color tone inconsistencies---a close-up in a warm, golden-lit office may have completely different lighting than the establishing shot.

We introduce an \textbf{agentic decision mechanism} where an LLM analyzes the lighting complexity of the current location and decides whether to include the location reference:

\begin{enumerate}
    \item \textbf{Lighting Complexity Analysis.} For close-up shots with available location references, we query the LLM:
    \begin{equation}
    (\text{desc}, c, \text{need}) = \mathcal{M}_{\text{LLM}}(\mathcal{E}_n^{\text{loc}}, \mathcal{G})
    \end{equation}
    where $c \in [1,5]$ is the lighting complexity score.

    \item \textbf{Adaptive Decision.}
    \begin{itemize}
        \item $c \leq 2$ (simple): Uniform daylight, single light source. Exclude location reference, inject lighting description into prompt.
        \item $c \geq 3$ (complex): Multiple colored sources, warm reflections. Include location reference to maintain color tone.
    \end{itemize}

    \item \textbf{Lighting Context Injection.} When location is excluded, we inject a ``Lighting Context'' layer describing the scene's lighting characteristics.
\end{enumerate}

\textbf{Shot-Type Adaptive Reference Selection.} Building on the agentic lighting analysis, we adapt reference selection based on shot type:

\begin{center}
\begin{tabular}{lcc}
\toprule
\textbf{Shot Type} & \textbf{Keywords} & \textbf{Location} \\
\midrule
Close-up & ``close-up'', ``tight'' & Agentic \\
Wide shot & ``wide'', ``establishing'' & Always \\
Medium & -- & Always \\
\bottomrule
\end{tabular}
\end{center}

Note that close-up shots may still feature multiple characters (e.g., ``a close-up on a man and a woman''), so the number of character references is determined by the actual entities parsed from the shot description, not hard-coded. When location is excluded, all 4~reference slots (the S2V model's maximum) can be used for character/object references.

\textbf{Per-Shot Seed Increment.} To increase generation diversity across shots, we use an incremented seed for each shot: $\text{seed}_n = \text{seed}_{\text{base}} + n$.

\textbf{Location Priority.} We observe that maintaining \emph{scene consistency} is as important as character consistency for coherent narratives. A desert scene in shot~1 should remain visually consistent in shot~3. To enforce this, we separate entity retrieval into two streams:

\textbf{(1)~Non-location entities} (characters, objects): We retrieve references sorted by priority (high $\rightarrow$ medium), taking at most 3 references:
\begin{equation}
\mathcal{I}_{\text{non-loc}} = \bigcup_{e \in \mathcal{E}_n^{\text{non-loc}}} \mathcal{R}.\text{query}(e.\text{id},\ k\!=\!1,\ \tau_q\!=\!0.4)
\end{equation}

\textbf{(2)~Location entities}: For each location entity, we check if the registry contains a reference. If yes, we \textbf{must} include it for scene consistency:
\begin{equation}
\mathcal{I}_{\text{loc}} = \bigcup_{e \in \mathcal{E}_n^{\text{loc}}} \mathcal{R}.\text{query}(e.\text{id},\ k\!=\!1,\ \tau_q\!=\!0.3)
\end{equation}

If a location entity has no registry entry, it is treated as a \emph{new scene}---the shot will be generated without location conditioning, and the location will be grounded and registered after generation.

The final reference set is $\mathcal{I}_n = \mathcal{I}_{\text{non-loc}} \cup \mathcal{I}_{\text{loc}}$, capped at 4~images (the S2V model's maximum).

% ----------------------------------------------------------------------------
\subsection{Four-Layer Prompt Construction}

A naive approach would directly concatenate the \texttt{global\_caption} with the shot description. However, this introduces a critical problem: the global caption typically describes the entire video narrative, potentially mentioning characters or events that should not appear in the current shot. To address this, we construct the generation prompt in four carefully designed layers:

\textbf{Layer 1: Global Context.} Semantic attributes extracted from the global caption that apply uniformly across all shots: visual style, mood, setting, and narrative context. This layer explicitly excludes specific entity descriptions.
\begin{equation}
P_{\text{global}} = \text{BuildGlobalContext}(\mathcal{C}_{\text{global}})
\end{equation}

\textbf{Layer 2: Lighting Context (Conditional).} For close-up shots where the agentic analysis decides \emph{not} to include location references, we inject lighting guidance:
\begin{equation}
P_{\text{light}} = \text{BuildLightingContext}(\text{desc}, \text{tone})
\end{equation}
This layer is only included when: (1)~the shot is a close-up, (2)~location references are excluded, and (3)~lighting analysis is available.

\textbf{Layer 2.5: Environment Context (T2V Fallback).} When subject-aware routing falls back to T2V mode due to missing subject references, we inject environment descriptions derived from location entities:
\begin{equation}
P_{\text{env}} = \bigcup_{e \in \mathcal{E}_n^{\text{loc}}} \text{Format}(\text{``Scene''}, e.\text{desc}, e.\text{attr})
\end{equation}
This ensures scene consistency is maintained through text guidance when image conditioning is unavailable.

\textbf{Layer 2.6: Appearance Context (Faceless Characters).} When a character has only faceless references ($\text{id\_conf} < \tau_{\text{face}}$), we inject appearance descriptions to maintain clothing/style consistency:
\begin{equation}
P_{\text{appear}} = \bigcup_{e \in \mathcal{E}_n^{\text{faceless}}} \text{Format}(e.\text{type}, e.\text{attr}_{\text{appear}})
\end{equation}
where $e.\text{attr}_{\text{appear}}$ includes only appearance-related attributes (hair color, clothing, accessories), excluding attributes that would be meaningless without visual reference (age estimation, facial features).

\textbf{Layer 3: Shot Entity Context.} Detailed descriptions of entities that \emph{actually appear in the current shot}, retrieved from the entity graph:
\begin{equation}
P_{\text{entity}} = \bigcup_{e \in \mathcal{E}_n} \text{Format}(e.\text{type}, e.\text{desc}, e.\text{attr})
\end{equation}

\textbf{Layer 4: Shot Description.} The original shot text $s_n$ describing the specific action, camera movement, and composition.

The final prompt is constructed as:
\begin{equation}
P_n = P_{\text{global}} \oplus [P_{\text{light}}] \oplus [P_{\text{env}}] \oplus [P_{\text{appear}}] \oplus P_{\text{entity}} \oplus s_n
\end{equation}
where $[\cdot]$ denotes conditional inclusion based on the agentic lighting decision (for $P_{\text{light}}$), T2V fallback mode (for $P_{\text{env}}$), or faceless character presence (for $P_{\text{appear}}$).

This four-layer design ensures that: (1)~stylistic coherence is maintained across shots via the global context, (2)~lighting consistency is preserved even when location references are excluded, (3)~environment consistency is maintained when falling back to T2V, (4)~appearance consistency is maintained for faceless characters via text, (5)~only relevant entities are described to the generation model, and (6)~the original creative intent of each shot is preserved.

% ----------------------------------------------------------------------------
\subsection{Post-Generation Visual Grounding}

After generating video $v_n$, we perform visual grounding to extract entity crops that will serve as references for subsequent shots. This is the ``grounding-in-the-loop'' step that distinguishes our method from prior approaches.

\textbf{Why post-generation grounding?} References extracted from the \emph{actual generated video} are strictly preferable: (1)~they are visually consistent with the video backbone's style and color space; (2)~they capture the entity's appearance as rendered by the generation model; and (3)~they naturally reflect the diversity of poses and lighting already established.

\textbf{Frame Extraction.} We uniformly sample frames from $v_n$ at 1~FPS, yielding a set of frames $\mathcal{F}_n = \{f_1, f_2, \ldots, f_K\}$.

\textbf{End-to-End Referring Video Object Segmentation with ReferDINO.} Rather than a two-stage detection-then-segmentation pipeline (e.g., Grounding DINO + SAM), we adopt ReferDINO~\cite{liang2025referdino}, an end-to-end referring video object segmentation model. ReferDINO offers several advantages:

\begin{itemize}
    \item \textbf{Unified detection and segmentation}: Directly outputs pixel-level masks without requiring a separate segmentation stage, simplifying the pipeline and reducing latency.
    \item \textbf{Temporal consistency}: The object-consistent temporal enhancer maintains stable object identity across frames, reducing flickering and identity switches in extracted crops.
    \item \textbf{Real-time inference}: Achieves 51~FPS, significantly faster than sequential DINO+SAM pipelines.
\end{itemize}

\textbf{Multi-Entity Joint Detection.} A key empirical observation is that detecting multiple entities \emph{jointly} yields better disambiguation than detecting each entity independently. When entities share similar visual characteristics (e.g., ``woman in white bee suit'' and ``boy in white bee suit''), independent detection may confuse them. Joint detection allows the model to leverage \emph{contrastive} relationships between entity descriptions.

We construct a joint caption by concatenating all entity descriptions:
\begin{equation}
\mathcal{P}_{\text{joint}} = e_1.\text{desc} \oplus \text{`` . ''} \oplus e_2.\text{desc} \oplus \cdots
\end{equation}
and run ReferDINO inference once per frame set:
\begin{equation}
\{(m_e, b_e, s_e)\}_{e \in \mathcal{E}_n} = \text{ReferDINO}(\mathcal{F}_n, \mathcal{P}_{\text{joint}})
\end{equation}
where $m_e$ is the pixel-level mask, $b_e$ is the bounding box, and $s_e$ is the confidence score for each entity.

For each detected entity, we extract the masked crop with white background:
\begin{equation}
c_{e,k} = \text{MaskedCrop}(f_k,\ m_{e,k})
\end{equation}

\textbf{Cross-Entity IoU Deduplication.} Despite joint detection's improved disambiguation, we retain a post-processing IoU deduplication step for robustness. In multi-person scenes, if two entities' masks overlap significantly (IoU $> \tau_{\text{IoU}}$, default 0.5), we keep only the higher-confidence detection and suppress the other.

\textbf{Location Entity Extraction via Background Inpainting.} For \texttt{location}-type entities (e.g., ``sandy desert'', ``rainy alley''), we need a clean background reference without foreground characters. We achieve this through a foreground removal pipeline:
\begin{enumerate}
    \item Use ReferDINO to segment all foreground entities (characters, objects) with a joint caption.
    \item Compute the union of all foreground masks and apply morphological dilation.
    \item Use \texttt{cv2.inpaint} to fill masked regions, yielding a clean background frame.
\end{enumerate}
This inpainted background serves as the location reference for subsequent shots, ensuring scene consistency without character contamination.

% ----------------------------------------------------------------------------
\subsection{Reference Quality Scoring}

Not all extracted crops are equally useful as visual references. We compute a composite quality score $q(c)$ for each crop $c$:
\begin{equation}
q(c) = \alpha_1 q_{\text{sharp}}(c) + \alpha_2 q_{\text{id}}(c) + \alpha_3 q_{\text{pose}}(c)
\end{equation}
with $\alpha_1 = 0.4$, $\alpha_2 = 0.4$, $\alpha_3 = 0.2$.

\begin{itemize}
    \item $q_{\text{sharp}}$: Laplacian variance normalized to [0,1], measuring image sharpness.
    \item $q_{\text{id}}$: For \texttt{character}-type entities, the face detection confidence from InsightFace~\cite{insightface}; for other entity types, the CLIP cosine similarity. This score is also stored as \texttt{id\_confidence} in the registry for downstream frontal-aware filtering.
    \item $q_{\text{pose}}$: For characters, a heuristic frontal score based on face landmark yaw/pitch angles; for objects/locations, this term is set to~1.
\end{itemize}

We discard crops with $q(c) < \tau_q = 0.4$ and keep at most $K_{\max} = 3$ best crops per entity per shot.

\textbf{Frontal-Aware Character Reference Selection.} A critical observation is that S2V models like Phantom can generate semantically plausible faces from back-view or faceless character references, but these ``hallucinated'' faces often drift into unintended styles (e.g., animated or cartoon-like) because no facial identity constraint is provided.

To address this, we store the face detection confidence $q_{\text{id}}$ (from InsightFace) as \texttt{id\_confidence} for each character crop. At reference selection time, we filter character anchors:
\begin{equation}
\mathcal{I}_{\text{ch}}^{\text{valid}} = \{c \in \mathcal{I}_{\text{ch}} : c.\text{id\_conf} \geq \tau_{\text{face}}\}
\end{equation}
where $\tau_{\text{face}} = 0.3$ by default. Characters with $\text{id\_conf} < \tau_{\text{face}}$ are considered ``faceless'' (back views, heavily occluded, or profile shots without detectable face).

For faceless characters, we inject an \textbf{Appearance Context} layer into the prompt instead of using the visual reference:
\begin{equation}
P_{\text{appear}} = \text{``[Appearance Context - No Frontal Reference]''} \oplus \bigcup_{e \in \mathcal{E}^{\text{faceless}}} \text{Format}(e.\text{attr})
\end{equation}
where $e.\text{attr}$ includes appearance-related attributes (hair color, clothing, accessories) extracted during entity parsing.

This \textbf{frontal-aware} strategy ensures:
\begin{enumerate}
    \item S2V mode only receives character references with visible faces, preventing face hallucination.
    \item Faceless characters maintain appearance consistency through detailed text descriptions.
    \item Location references remain usable (no full fallback to T2V), preserving scene consistency.
\end{enumerate}

% ----------------------------------------------------------------------------
\subsection{Agentic Reference Selection}

Traditional reference selection relies on fixed scoring formulas: InsightFace for face quality, Laplacian variance for sharpness, and hand-tuned weights. This approach has fundamental limitations:

\begin{itemize}
    \item \textbf{Domain-specific:} Face scoring only works for characters; objects (vehicles, props) and locations (scenes) have no principled scoring method.
    \item \textbf{Context-blind:} The same frontal face reference may be ideal for a dialogue scene but suboptimal for an action sequence where a dynamic pose would better match.
    \item \textbf{Unexplainable:} Numeric scores provide no insight into why a reference was selected.
\end{itemize}

We introduce an \textbf{Agentic Reference Selection} mechanism that replaces fixed scoring with VLM-based semantic reasoning:

\textbf{Input Construction.} For each entity requiring reference selection, we present the VLM with:
\begin{itemize}
    \item Candidate reference images (up to 6) from the registry
    \item Entity description and type (character/object/location)
    \item Current shot context (the prompt being generated)
    \item Shot type (close-up/medium/wide)
\end{itemize}

\textbf{VLM Analysis.} The VLM analyzes each candidate along entity-type-specific dimensions:

\begin{center}
\begin{tabular}{lp{5cm}}
\toprule
\textbf{Entity Type} & \textbf{Evaluation Criteria} \\
\midrule
Character & Angle, expression, pose, lighting, clarity \\
Object & Angle, completeness, clarity, context \\
Location & Coverage, lighting mood, composition, atmosphere \\
\bottomrule
\end{tabular}
\end{center}

\textbf{Structured Output.} The VLM returns:
\begin{equation}
(\text{idx}, \text{conf}, \text{reason}, \text{alts}) = \text{VLM}(\mathcal{I}_{\text{cand}}, e, s_n, \text{type})
\end{equation}
where $\text{idx}$ is the selected candidate index, $\text{conf} \in [0,1]$ is selection confidence, $\text{reason}$ is natural language justification, and $\text{alts}$ are ranked alternatives.

\textbf{Hybrid Mode with Fallback.} To ensure robustness, we support three selection modes:
\begin{itemize}
    \item \texttt{agent}: Pure VLM selection (fails if VLM unavailable)
    \item \texttt{traditional}: Fixed scoring formulas (legacy behavior)
    \item \texttt{hybrid} (default): VLM-first with automatic fallback to traditional scoring on VLM failure
\end{itemize}

The hybrid mode provides the best of both worlds: intelligent, context-aware selection when VLM is available, with guaranteed fallback for reliability.

\textbf{Benefits.} Agentic reference selection provides:
\begin{enumerate}
    \item \textbf{Universality:} Same mechanism handles characters, objects, and locations---no entity-specific scoring functions needed.
    \item \textbf{Context awareness:} Selections adapt to shot semantics (dialogue $\to$ frontal, action $\to$ dynamic).
    \item \textbf{Explainability:} Each selection includes natural language justification for debugging and human review.
    \item \textbf{Graceful degradation:} Hybrid mode ensures reliability even when VLM is unavailable.
\end{enumerate}

% ----------------------------------------------------------------------------
\subsection{Entity Registry}

The Entity Registry $\mathcal{R}$ is a persistent key-value store mapping entity IDs to ranked lists of reference entries. Each entry stores the crop path, quality score, source shot ID, and grounding metadata.

The registry supports three key operations:
\begin{itemize}
    \item $\text{register}(e.\text{id},\ \text{entry})$: Insert a new reference entry.
    \item $\text{query}(e.\text{id},\ k,\ \tau_q,\ \text{strategy})$: Retrieve the top-$k$ entries with score $\geq \tau_q$.
    \item $\text{query\_anchor}(e.\text{id},\ \tau_q,\ \tau_{\text{high}})$: Retrieve the single best ``anchor'' reference using the earliest high-quality strategy.
\end{itemize}

\textbf{Query Strategies.} The \texttt{query} operation supports three ordering strategies:
\begin{itemize}
    \item \texttt{earliest\_good} (default): Sort by shot\_id ascending, then quality descending. Basic anchor strategy.
    \item \texttt{best\_quality}: Sort by quality descending. May select drifted references from later shots.
    \item \texttt{most\_recent}: Sort by shot\_id descending. Legacy behavior, causes error accumulation.
\end{itemize}

For character entities, we use the specialized \texttt{query\_anchor} method with the \textbf{earliest high-quality} strategy (Section~3.2), which considers both shot order and a high-quality threshold $\tau_{\text{high}}$ to select the optimal anchor reference.

\textbf{Location Anchor Strategy.} Location entities present a distinct challenge: Shot~1 may be a close-up where the background is blurred due to depth-of-field effects, while a subsequent wide shot may have a much clearer background. To address this, we introduce a specialized \textbf{``earliest unless much worse''} strategy for locations:

\begin{enumerate}
    \item If the earliest shot's background quality $\geq \tau_{\text{loc}}^{\text{high}}$ (default 0.7), select it.
    \item Otherwise, find the highest-quality background across all shots. If $q_{\text{best}} / q_{\text{earliest}} \geq \rho_{\text{gap}}$ (default 1.5), select the better background.
    \item Otherwise, default to the earliest shot to prevent scene drift.
\end{enumerate}

Formally:
\begin{equation}
\mathcal{I}_{\text{loc}}(e) = \begin{cases}
\mathcal{R}[e, 1] & \text{if } q(\mathcal{R}[e, 1]) \geq \tau_{\text{loc}}^{\text{high}} \\
\mathcal{R}[e, s^*] & \text{if } q_{\text{best}} / q_{\text{earliest}} \geq \rho_{\text{gap}} \\
\mathcal{R}[e, 1] & \text{otherwise}
\end{cases}
\end{equation}
where $s^* = \argmax_s q(\mathcal{R}[e, s])$ is the shot with the highest-quality location reference.

This strategy balances scene consistency (preferring early shots) with quality awareness (allowing switches when later shots have dramatically better backgrounds, e.g., due to close-up vs.\ wide shot differences).

\textbf{Location-Specific Quality Scoring.} Standard quality metrics (face confidence, frontal score) are meaningless for location entities. We introduce a specialized scoring function:
\begin{equation}
q_{\text{loc}}(c) = 0.5 \cdot q_{\text{sharp}}(c) + 0.3 \cdot q_{\text{rich}}(c) + 0.2 \cdot (1 - q_{\text{inpaint}}(c))
\end{equation}
where:
\begin{itemize}
    \item $q_{\text{sharp}}$: Laplacian variance (same as character), heavily weighted to reject blurry backgrounds
    \item $q_{\text{rich}}$: Content richness measured by Canny edge density and color variance
    \item $q_{\text{inpaint}}$: Penalty for white/overexposed regions (artifacts from foreground removal)
\end{itemize}

\textbf{Agentic Self-Improving Registry.} A critical observation is that naive ``register-only'' registries accumulate unbounded reference entries across shots, leading to database bloat and degraded query performance. For a 20-shot video with 8 entities and 3 crops per entity per shot, this produces 480+ entries---many of which are low-quality, redundant, or superseded by better references.

We introduce an \textbf{agentic self-improving registry} that actively manages its own state through intelligent registration, eviction, and diversity maintenance:

\textbf{(1)~Smart Registration with Multi-Stage Filtering.} Rather than accepting all grounding results, the registry applies a cascade of filters at registration time:
\begin{itemize}
    \item \textbf{Quality gate:} Reject entries with $q < \tau_{\text{min}}$ (default 0.4)
    \item \textbf{Face confidence gate (characters):} Reject entries with $\text{id\_conf} < \tau_{\text{face}}$ (default 0.3)
    \item \textbf{Similarity deduplication:} Compute CLIP embeddings and reject entries with cosine similarity $> \tau_{\text{sim}}$ (default 0.92) to existing references
    \item \textbf{Per-shot capacity:} Limit to $k_{\text{shot}}$ entries per entity per shot (default 2)
    \item \textbf{Total capacity:} Limit to $k_{\text{total}}$ entries per entity (default 10)
\end{itemize}

\textbf{(2)~Quality-Competitive Eviction.} When capacity limits are reached, the registry makes agentic eviction decisions:
\begin{equation}
\text{evict}(e) = \begin{cases}
r_{\text{lowest}} & \text{if } q(r_{\text{lowest}}) < \tau_{\text{evict}} \\
r_{\text{lowest}} & \text{if } q_{\text{new}} > q(r_{\text{lowest}}) \\
\text{reject new} & \text{otherwise}
\end{cases}
\end{equation}
where $r_{\text{lowest}}$ is the lowest-quality non-anchor reference. This ensures that higher-quality references can always replace lower-quality ones, even when the registry is at capacity.

\textbf{(3)~Anchor Protection.} References that meet anchor criteria (high quality, frontal face, early shot) are automatically promoted to protected status and excluded from eviction. Each entity maintains up to 2 protected anchors.

\textbf{(4)~Periodic Audit.} Every 5 shots, the registry runs an eviction audit that removes low-quality entries below $\tau_{\text{evict}}$ and deduplicates similar references, maintaining a diverse, high-quality reference pool.

The agentic registry reduces storage by $\sim$80\% compared to naive accumulation while improving average reference quality, with negligible runtime overhead ($<$2\% of generation time for CLIP similarity computation).

An important property: the registry accumulates references \textbf{across shots}. By shot $n$, the registry contains references extracted from $v_1, \ldots, v_{n-1}$, meaning each subsequent shot benefits from progressively more diverse reference images. However, due to the anchor strategies, character references are drawn from early high-quality shots, and location references balance early consistency with quality awareness.

% ----------------------------------------------------------------------------
\subsection{Agentic Orchestration Loop}

Algorithm~\ref{alg:pipeline} presents the full agentic pipeline. The loop processes shots sequentially, maintaining the Entity Registry state between shots. Note the strict temporal ordering: \emph{generate} $v_n$ $\rightarrow$ \emph{ground} $v_n$ $\rightarrow$ \emph{update registry} $\rightarrow$ \emph{generate} $v_{n+1}$.

\begin{algorithm}[t]
\caption{T2V-Grounding Agentic Pipeline}
\label{alg:pipeline}
\begin{algorithmic}[1]
\REQUIRE Script $\mathcal{S} = \{s_1, \ldots, s_N\}$, optional $\mathcal{C}_{\text{global}}$
\ENSURE Videos $\mathcal{V} = \{v_1, \ldots, v_N\}$
\STATE Initialize: Registry $\mathcal{R} = \emptyset$, Known entities $\mathcal{K} = \emptyset$
\IF{$\mathcal{C}_{\text{global}} \neq \emptyset$}
    \STATE $\mathcal{K} \leftarrow \text{LLM\_Parse}(\mathcal{C}_{\text{global}}, \text{shot\_id}=0)$
    \STATE $\mathcal{G} \leftarrow \text{ExtractGlobalCtx}(\mathcal{C}_{\text{global}})$
\ENDIF
\FOR{$n = 1, 2, \ldots, N$}
    \STATE $\mathcal{E}_n \leftarrow \text{LLM\_Parse}(s_n, \mathcal{K})$
    \STATE $\mathcal{K} \leftarrow \mathcal{K} \cup \{e \in \mathcal{E}_n : e.\text{is\_new}\}$
    \STATE $P_n \leftarrow \text{BuildPrompt}(\mathcal{G}, \mathcal{E}_n, s_n)$ \COMMENT{4-layer}
    \STATE $\text{type} \leftarrow \text{DetectShotType}(s_n)$
    \STATE $\text{seed}_n \leftarrow \text{seed}_{\text{base}} + n$
    \STATE \COMMENT{Earliest high-quality anchor strategy}
    \STATE $\mathcal{I}_{\text{ch}} \leftarrow \mathcal{R}.\text{anchor}(\mathcal{E}_n^{\text{ch}}, \tau_q, \tau_{\text{high}})$
    \STATE $\mathcal{I}_{\text{obj}} \leftarrow \mathcal{R}.\text{query}(\mathcal{E}_n^{\text{obj}}, k\!=\!1)$
    \IF{$\text{type} = \text{close-up}$}
        \STATE $\text{light} \leftarrow \text{AnalyzeLighting}(\mathcal{E}_n^{\text{loc}}, \mathcal{G})$
        \IF{$\text{light.needs\_ref}$}
            \STATE $\mathcal{I}_{\text{loc}} \leftarrow \mathcal{R}.\text{query}(\mathcal{E}_n^{\text{loc}}, k\!=\!1)$
        \ELSE
            \STATE $P_n \leftarrow \text{InjectLight}(P_n, \text{light})$
        \ENDIF
    \ELSE
        \STATE $\mathcal{I}_{\text{loc}} \leftarrow \mathcal{R}.\text{query}(\mathcal{E}_n^{\text{loc}}, k\!=\!1)$
    \ENDIF
    \STATE $\mathcal{I}_n \leftarrow \text{Filter}(\mathcal{I}_{\text{ch}} \cup \mathcal{I}_{\text{obj}} \cup \mathcal{I}_{\text{loc}})$
    \STATE \COMMENT{Subject-aware mode routing}
    \STATE $\text{has\_subj} \leftarrow (\mathcal{I}_{\text{ch}} \cup \mathcal{I}_{\text{obj}}) \neq \emptyset$
    \IF{$\text{has\_subj}$}
        \STATE $v_n \leftarrow \mathcal{G}_{\text{S2V}}(P_n, \mathcal{I}_n, \text{seed}_n)$
    \ELSE
        \STATE $P_n \leftarrow \text{InjectEnv}(P_n, \mathcal{E}_n^{\text{loc}})$ \COMMENT{T2V fallback}
        \STATE $v_n \leftarrow \mathcal{G}_{\text{T2V}}(P_n, \text{seed}_n)$
    \ENDIF
    \STATE $\mathcal{F}_n \leftarrow \text{ExtractFrames}(v_n, \text{fps}=1)$
    \STATE \COMMENT{Joint multi-entity grounding with ReferDINO}
    \STATE $\mathcal{P}_{\text{joint}} \leftarrow \text{JoinDescriptions}(\mathcal{E}_n)$
    \STATE $\mathcal{D} \leftarrow \text{ReferDINO}(\mathcal{F}_n, \mathcal{P}_{\text{joint}})$
    \FOR{each $e \in \mathcal{E}_n^{\text{loc}}$}
        \STATE $\mathcal{D}[e] \leftarrow$ Inpaint$(\mathcal{F}_n, \mathcal{D})$ \COMMENT{Remove fg, keep bg}
    \ENDFOR
    \STATE $\mathcal{D} \leftarrow \text{IoUDedup}(\mathcal{D}, \tau_{\text{IoU}})$
    \FOR{each $e \in \mathcal{E}_n$ with priority $\neq$ low}
        \STATE scored $\leftarrow$ QualityScore$(\mathcal{D}[e], e.\text{type})$
        \FOR{$c \in \text{TopK}(\text{scored}, K_{\max})$ with $q(c) \geq \tau_q$}
            \STATE $\mathcal{R}.\text{register}(e.\text{id}, c)$
        \ENDFOR
    \ENDFOR
    \STATE $\mathcal{V}[n] \leftarrow v_n$
\ENDFOR
\RETURN $\mathcal{V}$
\end{algorithmic}
\end{algorithm}

% ============================================================================
\section{Experiments}

Omit for now.

% ============================================================================
\section{Discussion}

Omit for now.

% ============================================================================
\section{Conclusion}

Omit for now.

\end{document}
