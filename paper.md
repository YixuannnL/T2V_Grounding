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
Omit for now.
\end{abstract}

% ============================================================================
\section{Introduction}

The problem of generating a coherent multi-shot video from a text script is fundamentally different from single-shot video generation. A compelling visual narrative requires that the same character appearing across different shots---in different scenes, lighting conditions, camera angles, and actions---be recognizably the same person. Current state-of-the-art T2V models~\cite{wan2024wan,yang2024cogvideox,gen3} treat each shot as an independent generation task, resulting in significant character appearance drift that renders multi-shot narratives visually incoherent.

Several directions have been proposed to address visual consistency in generative models. Image-to-video (I2V) methods~\cite{wan2024wan} condition generation on a reference first frame, but this locks both the character appearance \emph{and} the composition/pose---unsuitable for shots with different camera angles or actions. Training-based approaches~\cite{dreambooth_video,dreamvideo} require per-character fine-tuning, which is impractical at inference time. Reference-conditioned generation approaches~\cite{ye2023ip,phantom} can condition on appearance without fixing pose, but they assume reference images are available as inputs---raising the question of where these references come from in an automated pipeline.

Our key observation is: \textbf{the generated video itself is the best source of reference images for future shots.} Rather than relying on an external character database or manual reference image provision, we propose to extract reference images \emph{from each generated shot} using visual grounding, and use them to condition subsequent shots. This creates a generation-grounding feedback loop that progressively builds a richer visual reference bank as more shots are generated.

We make the following contributions:
\begin{enumerate}
    \item A training-free \textbf{agentic pipeline} that maintains character visual consistency across multi-shot T2V generation without requiring external reference images.
    \item A \textbf{post-generation grounding} paradigm, where entity crops are extracted from generated videos (not assumed available beforehand) using open-vocabulary detection and segmentation.
    \item A \textbf{persistent Entity Registry} with Re-ID quality scoring that stores and retrieves the best visual references across shots.
    \item An \textbf{LLM-driven entity parser} that performs cross-shot coreference resolution, ensuring semantic entity identity (e.g., ``the detective'' in shot~3 = ``Alex'' from shot~1) is maintained throughout the script.
\end{enumerate}

% ============================================================================
\section{Related Work}

Omit for now.

\section{Method}

We introduce T2V-Grounding, an agentic pipeline for multi-shot video generation with cross-shot entity consistency. Given a multi-shot script $\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$ where each shot $s_n$ is a natural language description, and an optional global caption $\mathcal{C}_{\text{global}}$ describing the overall narrative context, our goal is to generate a sequence of videos $\mathcal{V} = \{v_1, v_2, \ldots, v_N\}$ such that entities appearing across multiple shots maintain consistent visual identity.

The pipeline consists of six components: (1)~an LLM-based Entity Parser, (2)~a Global Context Extractor, (3)~an Entity Registry, (4)~a Visual Grounding Module, (5)~a Reference Quality Scorer, and (6)~an Adaptive Video Generator. These are orchestrated by an agentic loop that processes shots sequentially. A key design is our \textbf{four-layer prompt construction}, which separates global semantic context, lighting guidance (for close-ups), shot-specific entity descriptions, and action text, preventing entity leakage across shots while maintaining stylistic and lighting coherence.

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

    \item \textbf{Post-Generation Verification.} After generating $v_n$, we sample $K$ frames uniformly and run person detection with NMS:
    \begin{equation}
    N_{\text{person}}^{\text{act}} = \text{mode}\bigl(\{\text{count}(\text{Det}(f_k))\}_{f_k \in \mathcal{F}}\bigr)
    \end{equation}
    where we take the mode (most frequent count) for robustness.

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

This verification loop ensures that the ``anchor'' references established in Shot~1 are semantically correct, preventing error accumulation.

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
