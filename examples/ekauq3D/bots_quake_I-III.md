# Bots for Quake I–III

Primary references: van Waveren, *The Quake III Arena Bot*, TU Delft, June 28 2001, Rev. 1 (§ = section, p = page), and the code of `id-Software/Quake-III-Arena` and `QW-Group/ktx`. Everything below was read in the research session (2026-10-23) unless listed under "Not verified".

## Survey

| Bot | Game | Author | Source | License | Navigation |
|---|---|---|---|---|---|
| Omicron | Quake | J.M.P. van Waveren ("Mr Elusive") | — | — | Drops waypoints ("bread crumbs") while wandering, no prior map knowledge (thesis §4.2.1) |
| Gladiator | Quake II | J.M.P. van Waveren | — | — | Precursor tech of Q3 bot; per-character fuzzy preferences (thesis §4.2.2) |
| Quake III bots (botlib + game AI) | Q3A | J.M.P. van Waveren | `id-Software/Quake-III-Arena` `code/botlib`, `code/game/ai_*.c` | GPL-2.0 | AAS areas + typed reachabilities, cached cluster routing |
| Frogbot (KTX port) | QuakeWorld | Robert "Frog" Field, Matt "asdf" McChesney, ParboiL | `QW-Group/ktx` `src/bot_*.c`, `route_*.c`, `marker_*.c` (C, converted from .qc 05/02/2016) | GPL-2.0 | Hand-placed markers, zones, flagged paths, precomputed travel times |
| Reaper Bot | Quake | Steven Polge, 1996 (v0.8 beta 11/15/96) | No public source; decompiled clones exist | Polge's notice restricts reuse in public mods | Learns levels while playing |
| FrikBot X | Quake | Ryan "Frika C" Smith | Zips in `Jason2Brownlee/QuakeBotArchive`; FrikBotNex (derived) states GPL | Original not verified | Hand-built waypoints; auto-waypoints while playing if none |

Other facts:
- Reaper: Guinness lists it as the first computer-controlled deathmatch opponent; Polge hired by Epic in 1997 (Wikipedia).
- Omicron and Gladiator were both written by van Waveren, who then took over the Q3 bots (fabiensanglard.net).
- `Jason2Brownlee/QuakeBotArchive` hosts original distributions of most Quake 1 bots; source zips exist for Eliminator, Frogbot CA/BotMatch, Reformed Reaper, Zeus, Termbot, GenBot, Darkbot and others. No license info on the page.

## Frogbot (KTX)

Path flags (`include/fb_globals.h`):

```
WATERJUMP_ (1<<1)   DM6_DOOR (1<<8)        ROCKET_JUMP (1<<9)     JUMP_LEDGE (1<<10, implies NO_DODGE)
VERTICAL_PLATFORM (1<<11)   BOTPATH_DOOR (1<<12)   BOTPATH_DOOR_CLOSED (1<<13)
REVERSIBLE (1<<14, bi-directional link in same zone)   WATER_PATH (1<<15)
DELIBERATE_AIR (1<<17)   WAIT_GROUND (1<<18)   STUCK_PATH (1<<19)   AIR_ACCELERATION (1<<20)
NO_DODGE (1<<21)   DELIBERATE_BACKUP (1<<22)   BOTPATH_CURLJUMP_HINT (1<<23)
BOTPATH_FULL_AIRCONTROL (1<<24)   BOTPATH_RJ_IN_PROGRESS (1<<25)
BOTPATH_KEEP_LINKED (1<<26, don't re-route on touching other markers)
```

- `route_calc.c`: after markers and zones/subzones are set, computes travel times between markers and goals; entry point `InitialiseMarkerRoutes()`. Rocket-jump paths: `time = 100000`, `rj_time = distance / (sv_maxspeed * (1 + max(0, dot(dir, hor_dir))))`.
- Per-map data: 78 `.bot` files in `resources/example-configs/ktx/bots/maps/` (dm2, dm3, dm4, dm6, aerowalk, ...). None for LibreQuake `lqdm*`.
- `.bot` format (plain text commands):

```
CreateMarker 152 -841 171
SetZone 1 7
SetMarkerPath 1 0 4
```

## Quake III bot (thesis + code)

### Architecture (§5, p21–23)

| Layer | Components |
|---|---|
| 4 | Team leader AI |
| 3 | Misc AI (fighting, obstacles, puzzles) · AI Network (FSM) · Commands |
| 2 | Fuzzy · Character · Goals · Navigation · Chats |
| 1 | Area Awareness System (all perception) · Basic Actions (output = player input) |

- Brain runs at 10 Hz (p22); movement AI every frame (§12, p67).
- Any query must finish in "several milliseconds" or the game hitches.
- `botlib` = layers 1–2 (server side); `game/ai_*.c` = layers 3–4.

### AAS: building areas (§6.2, p25–32)

1. Expand each solid brush by the player box: plane offset = `dot(v1, -n)`, v1 = box corner first touching the plane.
2. Add bevels: all axial planes through brush edges/corners not already present, plus axis-parallel planes along edges. Without bevels non-axial wedges over-expand.
3. CSG: drop contained brushes and overlapping parts.
4. One expanded brush set per hull size (stand, crouch) → one BSP. Liquid, teleporter, jump-pad, cluster-portal brushes compiled in as content.
5. BSP split heuristic: minimise number of convex leaves. Prune fully solid subtrees.
6. Portalise leaves → face boundary representation (shared faces/edges/vertices).
7. Flood from valid positions; delete unreached leaves (outside map).
8. Gravitational subdivision: split walkable areas with floor gaps by vertical planes through gap edges.
9. Merge adjacent areas while still convex and gap-free.

Area property: any two reachable points in an area are connected by a straight walk or swim.

Environment sampling (§6.3, p32–34):
- Point → area: descend BSP by plane side; on-plane goes front.
- Trace: recursive subdivision of the segment by BSP, front-to-back; collision at the first segment entering solid.
- Box → areas: expand entity box, descend both children when a plane splits it.

### Reachability (§6.4, p34–40)

Stored per reachability: start area, end area, start point, end point, travel time, face/edge crossed.

| Type | Rule |
|---|---|
| swim | Face between two swimmable areas |
| walk | Adjacent floors meet at an edge, or ground-edge E2 within step height of E1 (up or down, shallow water ok) |
| crouch | As walk, crouch required in either area |
| barrier jump | E2 above step height, ≤ max barrier height |
| walk off ledge | E2 more than step height below E1, or into swimmable water; box trace E1→E2 must be clear |
| water jump | A1 water, A2 dry/shallow; water-surface edge → ground edge ≤ max water-jump height |
| jump | Closest ground-edge points with gap(s) between; simulate full arc for obstacles; very close or slightly lower → walk off ledge instead |
| teleport | Trigger-flagged areas → destination area |
| elevator | Areas near bottom position → areas near top position only |
| jump pad | Trigger areas → landing area found by simulating flight |
| bobbing platform | Both directions between areas at end positions |
| rocket jump | From area centres only, to areas containing items only |

Code adds (`code/botlib/aasfile.h`): `TRAVEL_INVALID 1, WALK 2, CROUCH 3, BARRIERJUMP 4, JUMP 5, LADDER 6, WALKOFFLEDGE 7, SWIM 8, WATERJUMP 9, TELEPORT 10, ELEVATOR 11, ROCKETJUMP 12, BFGJUMP 13, GRAPPLEHOOK 14, DOUBLEJUMP 15, RAMPJUMP 16, STRAFEJUMP 17, JUMPPAD 18, FUNCBOB 19`.

Complexity: naive O(n²) area pairs; flood through faces for adjacency types; jump/rocket-jump candidates pruned by max jump distance/height and area bounds.

### Routing (§6.5, p40–45)

- Dynamic routing with cache (doors opened by buttons change routes); Dijkstra/A* on 5000+ areas too slow per query.
- Cache keyed by goal area (goals mostly fixed items): per source area → travel time + first reachability. Built on request.
- Two levels:
  - Area cache: BFS from goal over reversed reachabilities, inside the cluster plus touching portal areas.
  - Portal cache: BFS over portal areas; links = routes through clusters (from area caches).
- BFS preferred over Dijkstra: area sizes and reachability times are similar, sorting costs more than it saves (p43).
- Travel time includes crossing an area: per-area table from each incoming reachability end point to each outgoing reachability start point.
- Lookup: `min over portals p of cluster(time(src→p) + portal_cache(p→goal))`, and in-cluster time if goal in same cluster. Clusters usually have ≤10 portals.
- Memory: `M = TNA·NCP + Σ NAᵢ²`. Equal clusters: `M = TNA·NC − TNA + TNA²/NC`, minimum at `NC = √TNA`, `M ≈ 2·TNA·√TNA` (5000 areas: ~700k vs 25M).
- Cluster portals: auto-detected (door openings, windows) or hand-placed cluster-portal brushes. A portal separates exactly two clusters.
- Code: `be_aas_route.c` — `AAS_AreaRouteToGoalArea`, `AAS_AreaTravelTimeToGoalArea`, `AAS_UpdateAreaRoutingCache`, `AAS_UpdatePortalRoutingCache`, `AAS_EnableRoutingArea`, `AAS_PredictRoute`, `AAS_FreeOldestCache`, `AAS_WriteRouteCache`/`AAS_ReadRouteCache`.

Entities (§6.6, p46): entities linked into areas via a 2D linked list (entity ↔ area), so items per area and areas per entity are both cheap.

### Navigation (§12, p67–68)

- Next area: over the current area's reachabilities, pick the target area with the smallest travel time to the goal (not the cached first reachability, so avoided areas can be excluded).
- Per movement frame:
  1. On a mover (elevator/bobbing platform) → continue that reachability, or find one if landed there by accident.
  2. Airborne → airborne code for the current reachability type.
  3. Ground/swimming → keep last reachability unless area changed; else fetch next.
  4. Specialised movement code per reachability type.
- Same area as goal → straight line.
- Move in a direction (dodging), no routing: barrier → jump onto it; gap → predict jump, refuse into lava/slime/death pit; wall → move fails.

### Basic actions (§7, p47–48)

Attack, Use, Respawn, Jump, Crouch, Move Up, Move Down, Move Forward, Move Back, Move Left, Move Right, Walk, Talk, Gesture, `Move(dir, speed)` (independent of view), `View(angles)`, Select Weapon, Command, Say, Say Team.

Human limits (turn rate, reaction) are not in the interface; they are implemented in the AI (§7.1).

### Characteristics (§8.2, p50)

Mostly in [0, 1].

```
Attack skill:  <0.2 don't move | 0.2–0.4 forward/backward only | ≥0.4 circle strafe
               >0.7 random strafe direction change | >0.3 aim at enemy during retreat
Aim skill (per weapon):  <0.9 aim affected by enemy movement | 0.4–0.8 linear leading
               >0.8 exact movement leading | >0.6 splash at nearby geometry
               >0.5 prediction shots when enemy not visible
```

Others: name, gender, weapon weights file, view factor, view max change (deg/s), reaction time (s), aim accuracy (per weapon), chats file, characters per minute, chat tendencies, croucher, jumper, walker, weapon jumper, item weights file, aggression, self preservation, vengefulness, camper, easy fragger, alertness, fire throttle.

Characters are fixed during play; the aim is a bot slightly better than the player, not a perfect one (p51).

### Fuzzy weights (§9, p52–56)

Tree in C-like text; `case X` = branch while criterion < X; `default` = above last case; leaves `return` value. Evaluated recursively with interpolation between case boundaries → continuous output.

```
weight "Lightning Gun"
{
  switch(INVENTORY_LIGHTNING)
  {
    case 1: return 0;
    default:
    {
      switch(ENEMY_HORIZONTAL_DIST)
      {
        case 768:
        {
          switch(INVENTORY_LIGHTNINGAMMO)
          {
            case 1: return 0;
            case 50: return 70;
            case 100: return 77;
            case 200: return 80;
            default: return 80;
          }
        }
        case 800: return 0;
        default: return 0;
      }
    }
  }
}
```

- Highest weight wins (items and weapons). Absolute scale irrelevant, only balance.
- Genetic tuning (§9.4): N bots with the same trees duel; rank; pick two parents (high rank) and one child (low rank); child = average of parents' leaf values + random mutation; repeat with sped-up game time.
- Code: `be_ai_weight.c`, `be_ai_goal.c`, `be_ai_weap.c`, `be_ai_char.c`.

### Goals (§11, p65–66)

- LTG: fuzzy-chosen item, enemy (fight), item away from enemy (retreat), last seen enemy position (chase), camp spot; team: help/accompany teammate, defend key area, patrol, CTF flags.
- NBG (short term): items near the route without much detour; air while swimming.
- Goal finished when the item is seen gone; abandoned on timeout.

### Fighting (§13, p69–71)

- Acquire: 90° FOV enforced; fog and small apparent size reduce detection (alertness); invisible enemies not seen unless firing; may avoid a fight when weak or unseen and grab items first.
- Weapon: fuzzy weapon weights (range, ammo, inventory).
- Aim: instant-hit → on target (aim skill irrelevant); projectiles → lead per aim skill (none / linear / physics + geometry); splash at nearby geometry.
- Out of sight: predict the enemy's AAS shortest path back to the bot, fire splash at first visible point on it.
- Movement: per attack skill (static / back-forth / circle strafe / random strafe switch); jump/crouch per jumper/croucher; keep weapon-appropriate range; prefer height.

### Obstacles and puzzles (§14, p72–74)

- Static geometry is in AAS; only dynamic entities (players, doors, elevators, bars) are obstacles.
- Movement code traces forward and reports blocking to higher layers.
- Doors/bars compiled as flagged areas. Find activating buttons via target name; disable flagged areas for routing (and drop dependent caches) to tell reachable from unreachable buttons.
- Goal stack (LIFO) for chained buttons; predict route ahead and check flagged areas before bumping into them.

### AI network (§15, p75–78; `code/game/ai_dmnet.c`)

Nodes: Respawn, Stand, Seek_LTG, Seek_NBG, Seek_ActivateEntity, Battle_Fight, Battle_Chase, Battle_Retreat, Battle_NBG, Intermission, Observer.

- One node at a time; each think frame steps through nodes (via `AIEnter_*`) until one returns true.
- Death → Respawn → Seek_LTG. Game start / leave observer → Stand. Stand = typing chat.
- `AINode_Battle_Fight` order (Fig. 15.2): observer/intermission/dead checks → `BotFindEnemy` → no enemy → Seek_LTG → update last seen origin/area → `BotUpdateBattleInventory` → not visible → Chase or Seek_LTG → `BotBattleUseItems` → travel flags (`TFL_DEFAULT`, grapple, lava/slime, rocket jump) → `BotChooseWeapon` → `BotAttackMove` → on failure reset avoid reach → `BotAIBlocked` → `BotAimAtEnemy` → `BotCheckAttack` → `BotWantsToRetreat` → Retreat.

### Chats (§10, p57–64)

Synonym contexts, match templates with up to 8 variables, initial chats by type, Eliza-style reply chats with `&`/`!` keys and priorities, random string sets. Bot only chats when not in danger.

### Conclusions (§19, p90–92)

- AAS is computed automatically without human intervention (vs waypoint systems).
- Float round-off in large BSPs can break navigation; proposed fix: BSP per grid chunk, then combine.
- Routing caused some slowdown on some large maps, not enough to seriously degrade gameplay.
- Mostly stimulus–response; only planning is the button goal stack. Future: enemy path prediction, ambush spots, strategic location analysis.

## Pill engine integration points

- Output: `PlayerInput { forward, side, jump, yaw }` → `player_move(state, input, dt, world, settings)` — `engine/pill_assets/src/quake_player.rs:73`, `:251`. No crouch, no swim up/down.
- `MoveSettings::default()` (`quake_player.rs:43`): max_speed 320, accelerate 10, air_accelerate 10, air_wish_speed 30, friction 4, stop_speed 100, gravity 800, jump_speed 270, step_height 18, jump_buffer 0.15.
- Hull: `PLAYER_MINS [-16,-16,-24]`, `PLAYER_MAXS [16,16,32]`, `EYE_HEIGHT 22` (`quake_player.rs:12`). Single hull.
- World: `QuakeCollision::from_map` builds from `.map` brushes (`engine/pill_assets/src/quake_map.rs:1428`); no BSP. Queries: `trace_ray` (`quake_map.rs:1589`), `trace_player` (`quake_player.rs:105`).
- Existing path finding: grid A* + string pulling against the collision — `examples/ekauq3D/src/tour.rs:73`, `:161`.

## Not verified

- Contents of `be_aas_bspq3.c`, `be_ai_weight.c`.
- Numeric max barrier / water-jump heights (thesis: "fixed value in the game physics").
- License of the thesis text (author copyright on title pages). Q3 code is GPL-2.0: porting code (not ideas) makes the result GPL-2.0.
- Omicron, Gladiator, Eraser, 3ZB2, ACE, BrainWorks beyond what the thesis states.
- FrikBot X readme and QuakeWorld wiki (connection refused).

## Sources

- [The Quake III Arena Bot, van Waveren 2001 (mirror)](https://fabiensanglard.net/fd_proxy/quake3/The-Quake-III-Arena-Bot.pdf)
- [id-Software/Quake-III-Arena (GPL-2.0)](https://github.com/id-Software/Quake-III-Arena)
- [QW-Group/ktx (GPL-2.0)](https://github.com/QW-Group/ktx)
- [Fabien Sanglard – Quake 3 Source Code Review: AI](https://fabiensanglard.net/quake3/a.i.php)
- [Steve Polge – Wikipedia](https://en.wikipedia.org/wiki/Steve_Polge)
- [Randar's Bot Page – reviews (Reaper licensing)](https://www.quaddicted.com/files/webarchive/mrelusive.com_20170131/oldprojects/gladiator/reviews/Randar/Randar's%20Bot%20Page%20The%20Bot%20Reviews.htm)
- [QuakeBotArchive](https://github.com/Jason2Brownlee/QuakeBotArchive)
- [FrikBotNex listing](https://tomevault.io/tome/0xBrsm/FrikBotNex)
- [FrikBot X .way thread](https://forums.insideqc.com/viewtopic.php?p=7665)
- [Frogbot readme (Field credit)](https://se.aminet.net/game/patch/frogbot.readme)

## Later ML / RL / LLM work overlapping the Q3 bot

Researched 2026-10-23. "abstract" = arXiv abstract page read; "snippet" = search-result summaries only.

| Q3 bot component (thesis §) | Later work | Overlap | Checked |
|---|---|---|---|
| Whole bot, Q3A CTF (§15–17) | Jaderberg et al., *Human-level performance in first-person multiplayer games with population-based deep RL*, arXiv 1807.01281 (2018); Science 2019 ("FTW") | Same game/mode; pixels + game points only, no AAS or map knowledge | abstract |
| Genetic tuning of fuzzy weights (§9.4) | FTW: "two-tier optimisation", population of RL agents trained concurrently, each learns "its own internal reward signal" | Population-based selection over agent parameters vs duel → rank → crossover → mutate | abstract |
| 10 Hz brain vs per-frame movement (§5.1, §12) | FTW: "temporally hierarchical representation" | Two-timescale decisions | abstract |
| AI network / FSM + learned parts (§15) | Liu, Cann, Colbert, Saeedi, *Combining RL and Behavior Trees for NPCs in Video Games with AMD Schola*, arXiv 2510.14154 (Oct 2025) | Authored decision structure (BT) + RL skills; Unreal; RL still rare in commercial games | abstract |
| AAS reachability types + per-type movement (§6.4, §12.1) | Alonso, Peter, Goumard, Romoff (Ubisoft La Forge), *Deep RL for Navigation in AAA Video Games*, arXiv 2011.04764 (2020); IJCAI 2021 | Navmesh impractical with grapple/teleport/jump abilities; learns navigation for any ability; ≥90% success; maps an order of magnitude larger than typical deep RL; one modelled on Hyper Scape | abstract |
| AAS completeness / unreachable areas (§6.2 flood) | *Go-Explore Complex 3D Game Environments for Automated Reachability Testing*, arXiv 2209.00570 | Exploration vs navmesh to find reachable spots the navmesh misses | snippet |
| Human limits: reaction time, view max change, 90° FOV (§7.1, §8.2, §13.1) | Pearce & Zhu, *Counter-Strike Deathmatch with Large-Scale Behavioural Cloning*, arXiv 2104.04258 (NeurIPS 2021 workshop) | Human-like style learned from ~4M frames (5.5M / 95 h in other versions) instead of hand-set limits; medium built-in bot level | snippet |
| Human-likeness evaluation (§19.1) | BotPrize (Unreal Tournament): two teams passed 25% humanness with neuro-evolution + human-like action delays (ceur-ws Vol-1627) | Human-mimicking delays, cf. §7.1 | snippet |
| Characters, etiquette vs aggression (§8.2) | Sony AI GT Sophy (Nature 2022; QR-SAC); reward shaping aggression vs penalties; shipped GT7 Update 1.29, Sophy 2.0 in Spec II Nov 2023 (340 cars) | RL replacing a predictable rule-based AI in a shipped game | snippet |
| Chats: templates, Eliza replies, commands (§10, §16) | SIMA 2, Google DeepMind, arXiv 2512.04797 (Dec 2025): Gemini-based agent, converses, follows language + image instructions, generalises to unseen 3D games | Chat/command understanding and acting from one model | abstract |
| Team commands (§16–17) | Ubisoft *Teammates* (Nov 21 2025; Snowdrop; builds on Neo NPC, GDC 2024): voice-ordered squad NPCs; playtest a few hundred players | Natural-language orders vs template matching | snippet |
| Layer split: tick-rate control vs reasoning (§5.1) | KRAFTON *PUBG Ally* + NVIDIA ACE: on-device ASR + 2B SLM (reported Mistral-NeMo-Minitron-2B) + TTS, 8 GB VRAM GPUs; one report: BT for movement/combat at tick rate, SLM for intent/speech | Layers 1–2 vs 3–4 | snippet |
| Goal stack; "little or no planning" (§14.2, §19.3) | Voyager, arXiv 2305.16291 (NeurIPS 2023): GPT-4 Minecraft agent, automatic curriculum, code skill library, self-verification | LLM goal decomposition and skill reuse | snippet |
| Real-time control: general vs specialised model | Golchinfar, Vaziri, Marquardt, *Playing DOOM with 1.3M Parameters*, arXiv 2604.07385 (Apr 2026): 1.3M ModernBERT, ~31 ms/decision, ~31k human recordings; 178 frags vs 13 for all tested LLMs combined (Nemotron-120B, Qwen3.5-27B, GPT-4o-mini) | Per-frame layer | abstract |
| Same, earlier | de Wynter, *Will GPT-4 Run DOOM?*, arXiv 2403.05468: GPT-4 plays Doom from screenshot descriptions, no training | — | snippet |
| Bot-generated data | GameNGen, arXiv 2408.14837 (ICLR 2025): RL agent's Doom sessions train a diffusion model, 20 fps on one TPU. Microsoft Muse/WHAM (Nature 2025): ~7 years of Bleeding Edge play (>1B frames + controller actions) | Bots/player logs as world-model training data | snippet |

Searched, nothing found:
- 2024–2026 papers on learning jump / off-mesh links. Nearest: 2016 Utrecht MSc thesis (geometric jump-link generation for Recast navmeshes), and 2011.04764.
- An LLM playing Quake (only Doom).
- LLM-generated utility / fuzzy weights like §9. Only LLM → behavior tree work: ACM 10.1145/3696630.3728553, LLM-as-BT-Planner arXiv 2409.10444.

Sources:
- [Jaderberg et al. 2018, arXiv 1807.01281](https://arxiv.org/abs/1807.01281)
- [MIT Technology Review on FTW](https://www.technologyreview.com/2019/05/31/135146/deepminds-ai-has-used-teamwork-to-beat-humans-at-a-first-person-shooter/amp/)
- [AMD Schola RL+BT, arXiv 2510.14154](https://arxiv.org/abs/2510.14154)
- [Ubisoft La Forge DRL navigation, arXiv 2011.04764](https://arxiv.org/abs/2011.04764)
- [Ubisoft La Forge blog](https://ubisoft.com/en-us/studio/laforge/news/6bRtGllmfhuDqTHRS6KVLj/deep-reinforcement-learning-for-navigation-in-aaa-video-games)
- [Go-Explore reachability testing, arXiv 2209.00570](https://arxiv.org/pdf/2209.00570)
- [Pearce & Zhu, arXiv 2104.04258](https://arxiv.org/abs/2104.04258v2)
- [BotPrize / Makarov et al.](https://ceur-ws.org/Vol-1627/paper2.pdf)
- [Sony AI – GT Sophy](https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer)
- [GTPlanet – Sophy 5 years](https://www.gtplanet.net/?p=145563)
- [SIMA 2, arXiv 2512.04797](https://arxiv.org/abs/2512.04797)
- [DeepMind SIMA 2 blog](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)
- [Ubisoft Teammates](https://news.ubisoft.com/de-de/article/3mWlITIuWuu0MoVuR6o8ps/ubisoft-reveals-teammates--an-ai-experiment-to-change-the-game)
- [NVIDIA – PUBG Ally](https://developer.nvidia.com/blog/how-krafton-built-pubg-ally-a-co-playable-character-powered-by-nvidia-ace/)
- [KRAFTON press](https://press.krafton.com/KRAFTON-REVEALS-PLAYTEST-PLANS-FOR-PUBG-ALLY-BUILT-WITH-NVIDIA-ACE)
- [Voyager, arXiv 2305.16291](https://arxiv.org/abs/2305.16291v2)
- [Playing DOOM with 1.3M Parameters, arXiv 2604.07385](https://arxiv.org/abs/2604.07385)
- [Will GPT-4 Run DOOM?, arXiv 2403.05468](https://arxiv.org/abs/2403.05468)
- [GameNGen, arXiv 2408.14837](https://arxiv.org/abs/2408.14837v1)
- [Microsoft Muse](https://labs.ai.azure.com/innovations/muse/)
- [Utrecht jump-link thesis](https://studenttheses.uu.nl/handle/20.500.12932/22818)
- [LLM-as-BT-Planner, arXiv 2409.10444](https://arxiv.org/abs/2409.10444)
- [ACM LLM game BT](https://dl.acm.org/doi/10.1145/3696630.3728553)

## Plan: self-adapting, self-tuning bot for ekauq3D (M1 Pro, one weekend, one human tester)

Constraints: no hand-placed waypoints (per-map, data-driven); self-tuning weapons; moves = stand, run 3D, jump 3D, melee (axe), shotgun. Machine: Apple M1 Pro, 16 GB.

Decision: map adaptation from physics-sampled navigation (no ML); learning = evolutionary self-play tuning of a parameterised bot + statistics fitted from the human. Excluded: end-to-end RL (FTW Q3 CTF ≈ 450k games, MIT Technology Review); LLM in the control loop (DOOM 1.3M paper: all tested LLMs 13 frags vs 178).

### Codebase facts (read 2026-10-23)

- `player_move` (`engine/pill_assets/src/quake_player.rs:251`) is a pure function of `(PlayerState, PlayerInput, dt, QuakeCollision, MoveSettings)` → can run headless, faster than real time.
- Queries: `trace_box` (`quake_map.rs:1517`), `trace_ray` (`:1589`), `QuakeSpawn { classname, position, yaw, pitch }` (`:1705`).
- Weapons (`src/weapon.rs:20–44`): Axe range 64, refire 0.5 s; Shotgun 6 pellets, spread [0.04, 0.04], range 2048, refire 0.5 s.
- Missing: no health / damage / frag / respawn code in `examples/ekauq3D/src` (grep).

### Work items

| # | Work | Method | Precedent | M1 fit |
|---|---|---|---|---|
| 1 | Deathmatch rules: health, damage, death, respawn at `QuakeSpawn`, frags | code | — | hours |
| 2 | Headless match sim: N bots × `player_move` + `trace_ray` hits, fixed 1/72 s step, no render | code | thesis §9.4 time scaling | target ≥100× real time per core (unmeasured) |
| 3 | Auto nav graph per map | physics sampling | AAS §6.4 rules, measured not derived | seconds–minutes at map load (unmeasured) |
| 4 | Parameterised bot: Seek / Fight / Chase / Retreat (§15 subset), §13 aim/strafe, routing on #3 | code | thesis §8.2, §13, §15 | — |
| 5 | Self-tuning weapons/tactics via self-play | GA / CMA-ES | thesis §9.4; FTW PBT (arXiv 1807.01281) | overnight, rayon over cores |
| 6 | Human-likeness: log human `PlayerInput` + view angles per tick; fit turn-rate, reaction-time, aim-error distributions; clamp bot to them | statistics (too little data for BC) | Pearce & Zhu 2104.04258; BotPrize delays | minutes |
| 7 | Difficulty adaptation from human games | online update | thesis §8.2 skill thresholds | per game |

### #3 Nav graph (no waypoints)

1. Floor nodes: XY grid ~32 u, `trace_box` down with player hull, keep spots with headroom; flood from spawns, drop unreached (AAS §6.2 step 7).
2. Edges: node → neighbours within jump range; simulate `player_move` with a small input set (forward ± jump, few yaw offsets, run-up); keep sequences that land on target; store type (walk/jump/drop) + measured travel time. Exact w.r.t. the movement code even if it changes.
3. Routing: Dijkstra over nodes, per-goal cache (§6.5); clusters only if maps are large.
4. Per-map extras: danger map (deaths from self-play); node-to-node `trace_ray` visibility table for chase prediction (§13.2) and ambush spots.

### #5 Self-play tuning

- Genome (~20 floats): axe/shotgun switch distance, max fire distance, aim lead, aim error, strafe mode + switch rate (§8.2 attack skill), jump-dodge probability, retreat health, chase timeout, preferred range.
- Fitness: frag difference in round-robin duels across all maps (avoid overfitting one map).
- Optional weapon balance: evolve pellets / spread / refire toward a target (e.g. axe wins ≤ X% of fights < 64 u) — balance tuning, separate from bot tuning.

### #7 Difficulty adaptation

- Fix the evolved genome; scale only skill params (aim error, reaction time, view max change, strafe skill).
- After each human game step toward ~50% human win rate (frag-difference step). Human data too small for anything else.

### Dependencies

- #2 blocked by #1
- #4 blocked by #3
- #5 blocked by #2 AND #4
- #7 blocked by #5 AND #6

### Excluded

- RL movement/aim policy: compute + time (Ubisoft DRL nav, arXiv 2011.04764, AAA-scale).
- LLM per-frame control: arXiv 2604.07385.
- Full BSP AAS: sim-sampled graph (#3) gives reachability from the same movement code with less work.

### Unmeasured (decide whether #5 fits overnight)

- Headless sim speed.
- Node counts on `lqdm*` maps.
- Graph build time.
