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
    \item An \textbf{LLM-driven entity parser} that performs cross-shot coreference resolution, ensuring semantic entity identity (e.g., ``the detective'' in shot 3 = ``Alex'' from shot 1) is maintained throughout the script.
\end{enumerate}

% ============================================================================
\section{Related Work}

Omit for now.

\section{Method}

We introduce T2V-Grounding, an agentic pipeline for multi-shot video generation with cross-shot entity consistency. Given a multi-shot script $\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$ where each shot $s_n$ is a natural language description, and an optional global caption $\mathcal{C}_{\text{global}}$ that describes the overall narrative context, our goal is to generate a sequence of videos $\mathcal{V} = \{v_1, v_2, \ldots, v_N\}$ such that entities appearing across multiple shots maintain consistent visual identity.

The pipeline consists of six components: (1) an LLM-based Entity Parser, (2) a Global Context Extractor, (3) an Entity Registry, (4) a Visual Grounding Module, (5) a Reference Quality Scorer, and (6) an Adaptive Video Generator. These are orchestrated by an agentic loop that processes shots sequentially. A key design is our \textbf{four-layer prompt construction}, which separates global semantic context, lighting guidance (for close-ups), shot-specific entity descriptions, and action text, preventing entity leakage across shots while maintaining stylistic and lighting coherence.

% ----------------------------------------------------------------------------
\subsection{LLM-Based Entity Parser with Cross-Shot Coreference}

Before generating shot $n$, we run the Entity Parser to extract a structured entity list from the shot text $s_n$.

\textbf{Entity Extraction.} A language model $\mathcal{M}_{\text{LLM}}$ processes the shot description along with an accumulated context of known entities from previous shots:
\begin{equation}
\mathcal{E}_n = \mathcal{M}_{\text{LLM}}(s_n,\ \mathcal{K}_{n-1})
\end{equation}
where $\mathcal{K}_{n-1} = \{e \mid e \in \bigcup_{i<n} \mathcal{E}_i,\ e.\text{is\_new} = \texttt{true}\}$ is the set of all entities introduced in previous shots. The parser produces a list of entity records, where each entity $e$ contains:
\begin{itemize}
    \item $e.\text{entity\_id}$: a stable cross-shot identifier (e.g., \texttt{char\_alex}, \texttt{obj\_briefcase})
    \item $e.\text{type} \in \{\texttt{character}, \texttt{object}, \texttt{location}, \texttt{style}\}$
    \item $e.\text{text\_description}$: the natural language description used for grounding
    \item $e.\text{grounding\_priority} \in \{\texttt{high}, \texttt{medium}, \texttt{low}\}$
    \item $e.\text{is\_new}$: whether this entity first appears in the current shot
\end{itemize}

\textbf{Cross-Shot Coreference Resolution.} A critical challenge is that the same entity may be referred to differently across shots: ``Alex'' in shot 1 may appear as ``the detective'' in shot 3 or ``he'' in shot 4. We address this by providing the LLM with $\mathcal{K}_{n-1}$ as context and instructing it to assign the same \texttt{entity\_id} to coreferred entities. The LLM also maintains an alias list for each entity, enabling robust matching even under diverse referring expressions.

\textbf{Global Context Extraction.} In addition to entity extraction, the parser also extracts global semantic context from an optional \texttt{global\_caption} that describes the overall video narrative. This global context is decomposed into four dimensions:
\begin{itemize}
    \item $\text{visual\_style}$: Cinematographic style (e.g., ``cinematic, dramatic lighting'')
    \item $\text{mood}$: Emotional atmosphere (e.g., ``tense, suspenseful'')
    \item $\text{setting}$: Environmental description (e.g., ``sandy desolate desert'')
    \item $\text{narrative\_context}$: Story context (e.g., ``a standoff and pursuit sequence'')
\end{itemize}

Critically, the global context \textbf{excludes specific entity descriptions} to avoid injecting characters that should not appear in the current shot.

% ----------------------------------------------------------------------------
\subsection{Adaptive Video Generation with Mode Routing}

Given the entity list $\mathcal{E}_n$ and the Entity Registry state $\mathcal{R}_{n-1}$, we route shot $n$ to one of two generation modes:

\textbf{T2V Mode (Shot 1 or new entities).} When no visual references are available for any high-priority entity (i.e., the registry is empty or all relevant entities are new), we use a pure text-to-video model $\mathcal{G}_{\text{T2V}}$:
\begin{equation}
v_n = \mathcal{G}_{\text{T2V}}(s_n)
\end{equation}

\textbf{Shot 1 Entity Count Verification.} A critical observation is that T2V generation without reference constraints is prone to \emph{entity count errors}: if the prompt describes ``three men'', the model may generate four people. Since all subsequent shots depend on Shot 1's grounding results as anchors, such errors propagate and amplify throughout the entire multi-shot sequence.

To address this, we introduce a \textbf{Generation-Verification Loop} specifically for T2V mode (primarily Shot 1):

\begin{enumerate}
    \item \textbf{Entity Count Extraction.} During LLM parsing, we extract expected entity counts from the shot description. For each entity type $t$ (character, object), we compute:
    \begin{equation}
    N_t^{\text{expected}} = |\{e \in \mathcal{E}_n : e.\text{type} = t\}|
    \end{equation}

    \item \textbf{Post-Generation Verification.} After generating $v_n$, we sample $K$ frames uniformly and run person detection with NMS deduplication:
    \begin{equation}
    N_{\text{person}}^{\text{actual}} = \text{mode}(\{\text{count}(\text{Detect}(f_k)) : f_k \in \mathcal{F}_{\text{sample}}\})
    \end{equation}
    where we take the mode (most frequent count) across sampled frames for robustness.

    \item \textbf{Verification and Retry.} If $N_{\text{person}}^{\text{actual}} \neq N_{\text{person}}^{\text{expected}}$:
    \begin{itemize}
        \item Attempt 1-2: Retry with different random seed ($\text{seed}' = \text{seed} + 1000$)
        \item Attempt 3+: Use an \textbf{enhanced prompt} with explicit count constraint:
        \begin{equation}
        s_n' = \texttt{``[IMPORTANT: exactly } N \texttt{ people]''} \oplus s_n
        \end{equation}
    \end{itemize}

    \item \textbf{Termination.} Accept the generation if counts match, or after $M_{\text{max}}$ retries (default 3), warn the user and proceed with the best available result.
\end{enumerate}

This verification loop ensures that the ``anchor'' references established in Shot 1 are semantically correct, preventing error accumulation in subsequent shots.

\textbf{S2V Mode (subsequent shots with references).} When the registry contains reference crops for at least one high-priority entity in shot $n$, we switch to a subject-to-video model $\mathcal{G}_{\text{S2V}}$:
\begin{equation}
v_n = \mathcal{G}_{\text{S2V}}(s_n,\ \mathcal{I}_n)
\end{equation}
where $\mathcal{I}_n$ is the set of reference images retrieved from the registry.

\textbf{Why S2V and not I2V?} Image-to-video models treat the reference image as the first video frame, strongly constraining both appearance \emph{and} spatial layout/pose. This makes I2V unsuitable for shots with different camera angles or actions. S2V models like Phantom instead encode reference images as appearance tokens in a parallel conditioning stream, exerting appearance-level influence without fixing spatial structure.

\textbf{Reference Image Retrieval with Earliest High-Quality Anchor Strategy.} A critical observation is that \emph{appearance drift accumulates across shots}: the later a shot is generated, the more likely the character's face deviates from its original appearance. If we always use the most recent shot's grounding result as the reference, errors compound progressively.

A naive anchor strategy would simply select the earliest shot's reference. However, we observe that \textbf{the earliest shot is not always the best}: Shot 1 (typically a wide shot) may capture a character in profile or partial view, while Shot 2 (often a close-up) provides a clearer frontal face. Blindly selecting the earliest reference can lock in a poor-quality side profile.

To address this, we introduce an \textbf{``earliest high-quality''} anchor strategy that balances recency with quality:

\begin{enumerate}
    \item If the earliest shot contains a reference with quality $\geq \tau_{\text{high}}$ (default 0.85), select it.
    \item Otherwise, search all shots for the \emph{earliest} reference with quality $\geq \tau_{\text{high}}$.
    \item If no high-quality reference exists across all shots, fall back to the earliest shot's best reference (to still prevent drift accumulation).
\end{enumerate}

\begin{equation}
\mathcal{I}_{\text{anchor}}(e) = \begin{cases}
\mathcal{R}[e, \text{shot}=1] & \text{if } q(\mathcal{R}[e, \text{shot}=1]) \geq \tau_{\text{high}} \\
\arg\min_{\text{shot}} \{\mathcal{R}[e, \text{shot}] : q \geq \tau_{\text{high}}\} & \text{if } \exists \text{ high-quality ref} \\
\mathcal{R}[e, \text{shot}=1] & \text{otherwise (fallback)}
\end{cases}
\end{equation}

This strategy ensures that: (1) when a clear frontal face is captured early, it becomes the anchor; (2) when the earliest shot only has a poor-quality side profile but a later close-up provides a better reference, the system intelligently selects the higher-quality option; (3) ties are broken in favor of earlier shots to minimize drift.

\textbf{Agentic Light-Aware Close-up Strategy.} A naive approach excludes location references for all close-up shots to allow background defocus. However, this causes lighting and color tone inconsistencies---a close-up in a warm, golden-lit office may have completely different lighting than the establishing shot.

We introduce an \textbf{agentic decision mechanism} where an LLM analyzes the lighting complexity of the current location and decides whether to include the location reference:

\begin{enumerate}
    \item \textbf{Lighting Complexity Analysis.} For close-up shots with available location references, we query the LLM to analyze the scene's lighting characteristics:
    \begin{equation}
    (\text{lighting\_desc}, c_{\text{score}}, \text{needs\_ref}) = \mathcal{M}_{\text{LLM}}(\mathcal{E}_n^{\text{loc}}, \mathcal{G})
    \end{equation}
    where $c_{\text{score}} \in [1,5]$ is the lighting complexity score.

    \item \textbf{Adaptive Decision.}
    \begin{itemize}
        \item $c_{\text{score}} \leq 2$ (simple lighting): Uniform daylight, single light source. Text description is sufficient---exclude location reference, inject lighting description into prompt.
        \item $c_{\text{score}} \geq 3$ (complex lighting): Multiple colored sources, warm reflections, backlighting. Include location reference to maintain color tone consistency.
    \end{itemize}

    \item \textbf{Lighting Context Injection.} When location is excluded, we inject a ``Lighting Context'' layer into the prompt describing the scene's lighting characteristics, color tone, and depth-of-field guidance.
\end{enumerate}

This agentic approach ensures that close-ups maintain visual coherence with the overall scene while preserving the artistic intent of subject isolation.

\textbf{Shot-Type Adaptive Reference Selection.} Building on the agentic lighting analysis, we adapt reference selection based on shot type:

\begin{center}
\begin{tabular}{lcc}
\toprule
\textbf{Shot Type} & \textbf{Keywords} & \textbf{Location Decision} \\
\midrule
Close-up & ``close-up'', ``tight shot'' & Agentic (LLM decides) \\
Wide shot & ``wide shot'', ``establishing'' & Always include \\
Medium (default) & -- & Always include \\
\bottomrule
\end{tabular}
\end{center}

Note that close-up shots may still feature multiple characters (e.g., ``a close-up on a man and a woman''), so the number of character references is determined by the actual entities parsed from the shot description, not hard-coded. When location is excluded, all 4 reference slots (the S2V model's maximum) can be used for character/object references.

\textbf{Per-Shot Seed Increment.} To increase generation diversity across shots (avoiding the tendency for similar prompts with similar references to produce nearly identical outputs), we use an incremented seed for each shot: $\text{seed}_n = \text{seed}_{\text{base}} + n$.

\textbf{Reference Image Retrieval with Location Priority.} We observe that maintaining \emph{scene consistency} is as important as character consistency for coherent multi-shot narratives. A desert scene in shot 1 should remain visually consistent in shot 3, even if the camera angle changes. To enforce this, we separate entity retrieval into two streams:

\textbf{(1) Non-location entities} (characters, objects): We retrieve references sorted by grounding priority (high $\rightarrow$ medium), taking at most 3 references to reserve capacity for the location reference:
\begin{equation}
\mathcal{I}_{\text{non-loc}} = \bigcup_{e \in \mathcal{E}_n^{\text{non-loc}}} \mathcal{R}_{n-1}.\texttt{query}(e.\text{entity\_id},\ k=1,\ \tau_q=0.4)
\end{equation}

\textbf{(2) Location entities}: For each location entity in the current shot, we check if the registry contains a reference. If yes, we \textbf{must} include it to ensure scene consistency. We use a lower quality threshold ($\tau_q=0.3$) since background inpainting quality varies:
\begin{equation}
\mathcal{I}_{\text{loc}} = \bigcup_{e \in \mathcal{E}_n^{\text{loc}}} \mathcal{R}_{n-1}.\texttt{query}(e.\text{entity\_id},\ k=1,\ \tau_q=0.3)
\end{equation}

If a location entity has no registry entry, it is treated as a \emph{new scene}---the shot will be generated without location conditioning, and the location will be grounded and registered after generation for use in subsequent shots.

The final reference set is $\mathcal{I}_n = \mathcal{I}_{\text{non-loc}} \cup \mathcal{I}_{\text{loc|}|}$, capped at 4 images (the S2V model's maximum).

% ----------------------------------------------------------------------------
\subsection{Four-Layer Prompt Construction}

A naive approach would directly concatenate the \texttt{global\_caption} with the shot description. However, this introduces a critical problem: the global caption typically describes the entire video narrative, potentially mentioning characters or events that should not appear in the current shot. To address this, we construct the generation prompt in four carefully designed layers:

\textbf{Layer 1: Global Context.} Semantic attributes extracted from the global caption that apply uniformly across all shots: visual style, mood, setting, and narrative context. This layer explicitly excludes specific entity descriptions.
\begin{equation}
\text{prompt}_{\text{global}} = \texttt{BuildGlobalContext}(\mathcal{C}_{\text{global}})
\end{equation}

\textbf{Layer 2: Lighting Context (Conditional).} For close-up shots where the agentic analysis decides \emph{not} to include location references, we inject lighting guidance derived from the location analysis. This ensures close-ups maintain consistent lighting and color tone with the overall scene:
\begin{equation}
\text{prompt}_{\text{lighting}} = \texttt{BuildLightingContext}(\text{lighting\_desc}, \text{color\_tone})
\end{equation}
This layer is only included when: (1) the shot is a close-up, (2) location references are excluded, and (3) lighting analysis is available.

\textbf{Layer 3: Shot Entity Context.} Detailed descriptions of entities that \emph{actually appear in the current shot}, retrieved from the entity graph. For each entity $e \in \mathcal{E}_n$ with priority $\neq \texttt{low}$, we format its attributes as:
\begin{equation}
\text{prompt}_{\text{entity}} = \bigcup_{e \in \mathcal{E}_n} \texttt{FormatEntity}(e.\text{type}, e.\text{desc}, e.\text{attr})
\end{equation}

\textbf{Layer 4: Shot Description.} The original shot text $s_n$ describing the specific action, camera movement, and composition.

The final prompt is constructed as:
\begin{equation}
\text{prompt}_n = \text{prompt}_{\text{global}} \oplus [\text{prompt}_{\text{lighting}}] \oplus \text{prompt}_{\text{entity}} \oplus s_n
\end{equation}
where $[\cdot]$ denotes conditional inclusion based on the agentic lighting decision.

This four-layer design ensures that: (1) stylistic coherence is maintained across shots via the global context, (2) lighting consistency is preserved even when location references are excluded, (3) only relevant entities are described to the generation model, and (4) the original creative intent of each shot is preserved.

% ----------------------------------------------------------------------------
\subsection{Post-Generation Visual Grounding}

After generating video $v_n$, we perform visual grounding to extract entity crops that will serve as references for subsequent shots. This is the ``grounding-in-the-loop'' step that distinguishes our method from prior approaches.

\textbf{Why post-generation grounding?} References extracted from the \emph{actual generated video} are strictly preferable: (1) they are visually consistent with the video backbone's style and color space; (2) they capture the entity's appearance as rendered by the generation model; and (3) they naturally reflect the diversity of poses and lighting already established.

\textbf{Frame Extraction.} We uniformly sample frames from $v_n$ at 1 FPS, yielding a set of frames $\mathcal{F}_n = \{f_1, f_2, \ldots, f_K\}$.

\textbf{Open-Vocabulary Detection.} For each entity $e \in \mathcal{E}_n$ with priority $\neq \texttt{low}$, we run Grounding DINO~\cite{liu2023grounding} on each frame:
\begin{equation}
\mathcal{B}_{e,k} = \texttt{GDINO}(f_k,\ e.\text{text\_description},\ \tau_{\text{box}})
\end{equation}
For each detected box, we run SAM2~\cite{ravi2024sam2} to obtain a refined segmentation mask $m_{e,k}$, and extract the masked crop with white background:
\begin{equation}
c_{e,k} = \texttt{MaskedCrop}(f_k,\ m_{e,k})
\end{equation}

\textbf{Cross-Entity IoU Deduplication.} In multi-person scenes, open-vocabulary detection may incorrectly localize entity $A$'s bounding box onto entity $B$ (e.g., detecting ``young boy'' on an ``elderly man'' since both are persons). To address this, we perform cross-entity IoU deduplication within each frame:

\begin{enumerate}
    \item Collect all detection results across all entities for each frame.
    \item Sort detections by confidence score (descending).
    \item For each detection, suppress any lower-confidence detection from a \emph{different} entity with IoU $> \tau_{\text{IoU}}$ (default 0.5).
\end{enumerate}

This ensures that each detected region is assigned to at most one entity---the one with highest detection confidence.

\textbf{Location Entity Extraction via Background Inpainting.} For \texttt{location}-type entities (e.g., ``sandy desert'', ``rainy alley''), we need a clean background reference without foreground characters or objects. We achieve this through a foreground removal pipeline:
\begin{enumerate}
    \item Detect all foreground elements using a broad query (``person . people . man . woman . object . bag . item'') with Grounding DINO.
    \item For each detected box, obtain a precise segmentation mask via SAM2.
    \item Compute the union of all foreground masks and apply morphological dilation to avoid edge artifacts.
    \item Use \texttt{cv2.inpaint} (Telea algorithm) to fill the masked foreground regions, yielding a clean background frame.
\end{enumerate}
This inpainted background serves as the location reference for subsequent shots, ensuring scene consistency without character contamination.

% ----------------------------------------------------------------------------
\subsection{Reference Quality Scoring}

Not all extracted crops are equally useful as visual references. We compute a composite quality score $q(c)$ for each crop $c$:
\begin{equation}
q(c) = \alpha_1 \cdot q_{\text{sharp}}(c) + \alpha_2 \cdot q_{\text{id}}(c) + \alpha_3 \cdot q_{\text{pose}}(c)
\end{equation}
with $\alpha_1 = 0.4$, $\alpha_2 = 0.4$, $\alpha_3 = 0.2$.

\begin{itemize}
    \item $q_{\text{sharp}}$: Laplacian variance normalized to [0,1], measuring image sharpness.
    \item $q_{\text{id}}$: For \texttt{character}-type entities, the face detection confidence from InsightFace~\cite{insightface}; for other entity types, the CLIP cosine similarity.
    \item $q_{\text{pose}}$: For characters, a heuristic frontal score based on face landmark yaw/pitch angles; for objects/locations, this term is set to 1.
\end{itemize}

We discard crops with $q(c) < \tau_q = 0.4$ and keep at most $K_{\text{max}} = 3$ best crops per entity per shot.

% ----------------------------------------------------------------------------
\subsection{Entity Registry}

The Entity Registry $\mathcal{R}$ is a persistent key-value store mapping entity IDs to ranked lists of reference entries. Each entry stores the crop path, quality score, source shot ID, and grounding metadata.

The registry supports three key operations:
\begin{itemize}
    \item $\texttt{register}(e.\text{entity\_id},\ \text{entry})$: Insert a new reference entry.
    \item $\texttt{query}(e.\text{entity\_id},\ k,\ \tau_q,\ \text{strategy})$: Retrieve the top-$k$ entries with score $\geq \tau_q$, ordered by the specified strategy.
    \item $\texttt{query\_anchor}(e.\text{entity\_id},\ \tau_q)$: Retrieve the single best ``anchor'' reference---the earliest high-quality entry.
\end{itemize}

\textbf{Query Strategies.} The \texttt{query} operation supports three ordering strategies:
\begin{itemize}
    \item \texttt{earliest\_good} (default): Sort by \texttt{shot\_id} ascending, then quality descending. Basic anchor strategy.
    \item \texttt{best\_quality}: Sort by quality descending. May select drifted references from later shots.
    \item \texttt{most\_recent}: Sort by \texttt{shot\_id} descending. Legacy behavior, causes error accumulation.
\end{itemize}

For character entities, we use the specialized \texttt{query\_anchor} method with the \textbf{earliest high-quality} strategy (Section 3.2), which considers both shot order and a high-quality threshold $\tau_{\text{high}}$ to select the optimal anchor reference.

An important property: the registry accumulates references \textbf{across shots}. By shot $n$, the registry contains references extracted from $v_1, \ldots, v_{n-1}$, meaning each subsequent shot benefits from progressively more diverse reference images. However, due to the anchor strategy, character references are always drawn from early shots to maintain appearance fidelity.

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
    \STATE $\mathcal{G} \leftarrow \text{ExtractGlobalContext}(\mathcal{C}_{\text{global}})$
\ENDIF
\FOR{$n = 1, 2, \ldots, N$}
    \STATE $\mathcal{E}_n \leftarrow \text{LLM\_Parse}(s_n, \mathcal{K})$
    \STATE $\mathcal{K} \leftarrow \mathcal{K} \cup \{e \in \mathcal{E}_n : e.\text{is\_new}\}$
    \STATE $\text{prompt}_n \leftarrow \text{BuildPrompt}(\mathcal{G}, \mathcal{E}_n, s_n)$ \COMMENT{4-layer}
    \STATE $\text{shot\_type} \leftarrow \text{DetectShotType}(s_n)$ \COMMENT{close-up/wide/medium}
    \STATE $\text{seed}_n \leftarrow \text{seed}_{\text{base}} + n$ \COMMENT{per-shot seed increment}
    \STATE \COMMENT{Retrieve refs with earliest high-quality anchor strategy}
    \STATE $\mathcal{I}_{\text{char}} \leftarrow \mathcal{R}.\text{query\_anchor}(\mathcal{E}_n^{\text{char}}, \tau_q=0.4, \tau_{\text{high}}=0.85)$
    \STATE $\mathcal{I}_{\text{obj}} \leftarrow \mathcal{R}.\text{query}(\mathcal{E}_n^{\text{obj}}, k=1, \tau_q=0.4, \text{earliest})$
    \IF{$\text{shot\_type} = \text{close-up}$}
        \STATE $\text{light\_analysis} \leftarrow \text{AnalyzeCloseupLighting}(\mathcal{E}_n^{\text{loc}}, \mathcal{G})$
        \IF{$\text{light\_analysis.needs\_ref} = \texttt{true}$}
            \STATE $\mathcal{I}_{\text{loc}} \leftarrow \mathcal{R}.\text{query}(\mathcal{E}_n^{\text{loc}}, k=1, \tau_q=0.3)$
        \ELSE
            \STATE $\text{prompt}_n \leftarrow \text{InjectLightingContext}(\text{prompt}_n, \text{light\_analysis})$
        \ENDIF
    \ELSIF{$\text{shot\_type} \neq \text{close-up}$}
        \STATE $\mathcal{I}_{\text{loc}} \leftarrow \mathcal{R}.\text{query}(\mathcal{E}_n^{\text{loc}}, k=1, \tau_q=0.3, \text{earliest})$
    \ENDIF
    \STATE $\mathcal{I}_n \leftarrow \text{FilterByMaxRefs}(\mathcal{I}_{\text{char}} \cup \mathcal{I}_{\text{obj}} \cup \mathcal{I}_{\text{loc}}, \text{shot\_type})$
    \IF{$\mathcal{I}_n \neq \emptyset$}
        \STATE $v_n \leftarrow \mathcal{G}_{\text{S2V}}(\text{prompt}_n, \mathcal{I}_n, \text{seed}_n)$
    \ELSE
        \STATE $v_n \leftarrow \mathcal{G}_{\text{T2V}}(\text{prompt}_n, \text{seed}_n)$
    \ENDIF
    \STATE $\mathcal{F}_n \leftarrow \text{ExtractFrames}(v_n, \text{fps}=1)$
    \STATE \COMMENT{Ground all entities, then cross-entity IoU dedup}
    \STATE $\mathcal{D}_{\text{all}} \leftarrow \emptyset$
    \FOR{each $e \in \mathcal{E}_n$ with priority $\neq$ low}
        \IF{$e.\text{type} = \texttt{location}$}
            \STATE $\mathcal{D}_{\text{all}}[e] \leftarrow$ BackgroundInpaint$(\mathcal{F}_n)$
        \ELSE
            \STATE $\mathcal{D}_{\text{all}}[e] \leftarrow$ GDINO+SAM2$(\mathcal{F}_n, e.\text{text\_desc})$
        \ENDIF
    \ENDFOR
    \STATE $\mathcal{D}_{\text{dedup}} \leftarrow \text{CrossEntityIoUDedup}(\mathcal{D}_{\text{all}}, \tau_{\text{IoU}}=0.5)$
    \FOR{each $e \in \mathcal{E}_n$ with priority $\neq$ low}
        \STATE scored $\leftarrow$ QualityScore$(\mathcal{D}_{\text{dedup}}[e], e.\text{type})$
        \FOR{$c \in \text{TopK}(\text{scored}, K_{\text{max}})$ with $q(c) \geq \tau_q$}
            \STATE $\mathcal{R}.\text{register}(e.\text{entity\_id}, c)$
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
