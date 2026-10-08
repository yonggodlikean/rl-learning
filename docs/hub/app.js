/**
 * RL 学习中枢 (RL Learning Hub) - 前端核心逻辑
 * 纯本地静态驱动 · 目录随仓库自然生长 · 零外部依赖
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. 常量与可选教学导读批注 (Optional Annotations)
  // =========================================================================

  const CATALOG_URL = './catalog.json';
  const STORAGE_KEY = 'rl_hub_state_v1';
  const NOTE_LAN_ROOT = 'http://10.147.19.126:8898/';

  /**
   * 可选导读批注数据：仅作为对应章节内的重点导读补充，严禁作为封闭课程体系。
   * 显式绑定 chapterId（mrp -> easyrl-2.1-2.2.2, mdp -> easyrl-2.3），题目数与标题动态自 catalog 解析。
   */
  const KNOWLEDGE_TOPICS = [
    {
      id: 'mrp-return',
      chapterId: 'easyrl-2.1-2.2.2',
      defaultChapterTitle: '马尔可夫奖励过程 (MRP)',
      title: '折扣回报与倒序递推',
      textbookRef: 'EasyRL §2.1–§2.2.2',
      formulaText: 'G_t = r_{t+1} + γ · G_{t+1}',
      formulaHtml: 'G<sub>t</sub> = r<sub>t+1</sub> + γ · G<sub>t+1</sub>',
      formulaExpandedHtml: '展开式: G<sub>t</sub> = r<sub>t+1</sub> + γ r<sub>t+2</sub> + γ<sup>2</sup> r<sub>t+3</sub> + ... + γ<sup>T-t-1</sup> r<sub>T</sub>',
      conceptDiff: '即时奖励 r_{t+1}（单步标量反馈） vs 累计回报 G_t（未来衰减奖励总和）',
      intuition: '从终止步 T 逆向倒序递推，前一步回报可便捷复用后一步已算出的后续回报；求标量 G_0 时，前向加权求和与倒序递推均可在 O(T) 内完成。',
      noteSource: 'RL-book',
      notePath: '马尔可夫奖励过程 / 回报',
      checkpointQuestions: ['q05', 'q06'],
      codeExercise: {
        q: 'q11',
        funcName: 'compute_return',
        desc: '接收 rewards 数组与折扣因子 gamma，倒序递推返回标量 G_0。'
      },
      reflectionPrompt: '为什么最后一个奖励是 1，最前面的回报会是 0.6561？',
      staticCalc: {
        title: '具象数值静态推演 (Static Hand-Calculation)',
        badge: '静态数据流',
        rows: [
          { label: '输入奖励序列', val: 'rewards = [0, 0, 0, 0, 1], γ = 0.9 (步长 T=5)' },
          { label: '倒序递推步 G_4', val: 'r_5 = 1.0' },
          { label: '倒序递推步 G_3', val: '0 + 0.9 × 1.0 = 0.9' },
          { label: '倒序递推步 G_2', val: '0 + 0.9 × 0.9 = 0.81' },
          { label: '倒序递推步 G_1', val: '0 + 0.9 × 0.81 = 0.729' },
          { label: '起始标量回报 G_0', val: '0 + 0.9 × 0.729 = 0.6561 (即 0.9⁴)', highlight: true }
        ],
        nuance: '注：倒序递推能便捷复用后续回报；返回值为初始步的标量回报 G_0。前向加权累加亦可在 O(T) 时间内求得。'
      }
    },
    {
      id: 'mrp-bellman',
      chapterId: 'easyrl-2.1-2.2.2',
      defaultChapterTitle: '马尔可夫奖励过程 (MRP)',
      title: 'MRP 与贝尔曼备份',
      textbookRef: 'EasyRL §2.1–§2.2.2',
      formulaText: 'V = R + γ · P · V',
      formulaHtml: 'V = R + γ · P · V',
      formulaExpandedHtml: '标量形式: V(s) = R(s) + γ ∑<sub>s\'</sub> P(s\'|s) V(s\')',
      conceptDiff: '单条轨迹采样回报样本 G_t vs 状态期望价值函数 V(s)',
      intuition: '当前状态的真实价值由离开该状态收到的期望即时奖励与按转移概率加权的下一状态预期价值折现之和构成。',
      noteSource: 'RL-book',
      notePath: '贝尔曼方程',
      checkpointQuestions: ['q07', 'q08', 'q09'],
      codeExercise: {
        q: 'q12',
        funcName: 'bellman_backup',
        desc: '单轮同步 MRP 备份：利用奖励向量 R、转移矩阵 P、折现因子 gamma 与旧值向量 V，计算新价值向量。'
      },
      reflectionPrompt: '为什么一轮里所有新 V 都必须使用同一份旧 V？',
      staticCalc: {
        title: '具象数值静态推演 (Static Hand-Calculation)',
        badge: '静态数据流',
        rows: [
          { label: '状态规格', val: '2 状态系统 | R = [1.0, 0.0] | γ = 0.9' },
          { label: '转移概率 P', val: '[[0.5, 0.5], [0.0, 1.0]]' },
          { label: '旧价值向量 V', val: 'V_old = [2.0, 4.0]' },
          { label: '期望折现 P @ V', val: 'P @ [2.0, 4.0] = [3.0, 4.0]' },
          { label: '备份新价值 V_new', val: 'R + 0.9 × [3.0, 4.0] = [1.0 + 2.7, 0.0 + 3.6] = [3.7, 3.6]', highlight: true }
        ],
        nuance: '数学考量：本练习严格遵循“同步备份（Synchronous Sweep）”惯例，所有状态计算均基于上一轮旧 V_old；学术上异步就地更新（In-place / Asynchronous）同样可能有效，但非当前检查点判分基准。'
      }
    },
    {
      id: 'mdp-vq',
      chapterId: 'easyrl-2.3',
      defaultChapterTitle: '马尔可夫决策过程 (MDP)',
      title: 'V、Q 与策略加权',
      textbookRef: 'EasyRL §2.3',
      formulaText: 'V^π(s) = ∑_{a} π(a|s) · Q^π(s, a)',
      formulaHtml: 'V<sup>π</sup>(s) = ∑<sub>a ∈ A</sub> π(a|s) · Q<sup>π</sup>(s, a)',
      formulaExpandedHtml: '精确定义: V<sup>π</sup>(s) = ∑<sub>a</sub> π(a|s) Q<sup>π</sup>(s, a)；单步备份算子: V<sub>new</sub>(s) = ∑<sub>a</sub> π(a|s) [ R(s, a) + γ ∑<sub>s\'</sub> P(s\'|s, a) V<sub>old</sub>(s\') ]',
      conceptDiff: '动作价值 Q(s,a)（已选定动作的分支价值） vs 状态价值 V(s)（按策略概率分布求综合期望）',
      intuition: '在状态 s 下有多个动作分支，V(s) 表达的是遵循当前策略分布 π 时所有可选分支动作收益的期望平均。',
      noteSource: 'RL-book',
      notePath: '马尔可夫决策过程',
      checkpointQuestions: ['q02', 'q05', 'q06', 'q07'],
      codeExercise: {
        q: 'q11',
        funcName: 'policy_backup',
        desc: '单轮固定策略期望备份：依据转移概率、奖励、策略矩阵与旧价值，执行单轮期望加权更新。'
      },
      reflectionPrompt: 'Q 已固定动作 a，为什么 V 还要按策略加权？',
      staticCalc: {
        title: '具象数值静态推演 (Static Hand-Calculation)',
        badge: '静态数据流',
        rows: [
          { label: '候选动作价值 Q', val: 'Q(s, a₁) = 2.0,  Q(s, a₂) = 6.0' },
          { label: '当前策略分布 π', val: 'π(a₁|s) = 0.75, π(a₂|s) = 0.25' },
          { label: '策略期望价值 V', val: 'V = 0.75 × 2.0 + 0.25 × 6.0 = 1.5 + 1.5 = 3.0', highlight: true },
          { label: '对比贪心极值 max Q', val: '若取 max_a Q(s, a) = 6.0 (贪心上界与期望均值的本质区别)' }
        ],
        nuance: '数学考量：理论真值满足严格关系 V^π(s) = ∑_a π(a|s) Q^π(s, a)。而在单轮备份算法中，则是基于上一轮 V_old 计算单步候选 Q_old 并加权更新为 V_new，需明确区分理论精确恒等式与单步备份算子。'
      }
    },
    {
      id: 'mdp-dp',
      chapterId: 'easyrl-2.3',
      defaultChapterTitle: '马尔可夫决策过程 (MDP)',
      title: '策略迭代与价值迭代',
      textbookRef: 'EasyRL §2.3',
      formulaText: 'V*(s) = max_{a} [ R(s, a) + γ ∑_{s\'} P(s\'|s, a) V*(s\') ]',
      formulaHtml: 'V*(s) = max<sub>a ∈ A</sub> [ R(s, a) + γ ∑<sub>s\'</sub> P(s\'|s, a) V*(s\') ]',
      formulaExpandedHtml: '精确贝尔曼最优方程两端一致取 V*；单步备份与贪心动作: V<sub>new</sub>(s) = max<sub>a</sub> Q<sub>old</sub>(s, a), a<sub>greedy</sub>(s) = argmax<sub>a</sub> Q<sub>old</sub>(s, a) (单轮贪心动作非全局最优策略 π*)',
      conceptDiff: '策略评估（按策略期望线性加权） vs 最优备份（直接选取贪心最大值 max_a）',
      intuition: '策略评估度量既有策略的真实好坏；最优备份越过当前策略分布，直接以未来最强动作为基准收敛到最优价值函数。',
      noteSource: '强化学习核心导引',
      notePath: '贝尔曼方程',
      checkpointQuestions: ['q08', 'q10'],
      codeExercise: {
        q: 'q12',
        funcName: 'optimal_backup',
        desc: '执行单轮贝尔曼最优备份：返回 (new_values, greedy_actions)，若存在最大值平局，优先选取首个出现动作。'
      },
      reflectionPrompt: '策略评估取加权平均，最优备份为何取 max？',
      staticCalc: {
        title: '具象数值静态推演 (Static Hand-Calculation)',
        badge: '静态数据流',
        rows: [
          { label: '动作 a₁ 备份值', val: 'R(s, a₁) + γ ∑ P V = 4.2' },
          { label: '动作 a₂ 备份值', val: 'R(s, a₂) + γ ∑ P V = 5.8' },
          { label: '动作 a₃ 备份值', val: 'R(s, a₃) + γ ∑ P V = 5.8 (产生平局)' },
          { label: '最优价值 V_new', val: 'max(4.2, 5.8, 5.8) = 5.8', highlight: true },
          { label: '单轮贪心动作 a_greedy', val: '选取首个达到极大值的动作: a₂ (依题库惯例决胜；单轮贪心动作非最终最优策略 π*)' }
        ],
        nuance: '代码边界：本题仅为单轮最优算子备份（V_new = max_a Q_old），所得单轮贪心动作并非全局最优策略 π*，亦非包含收敛判定与外层循环的完整求解器。'
      }
    }
  ];

  /**
   * 手工保留的次级学习资源
   */
  const VERIFIED_RESOURCES = [
    {
      id: 'res-rl-book',
      type: '局域网个人笔记',
      title: 'RL-book',
      desc: '教材阅读笔记，涵盖马尔可夫奖励过程基础定义、累计回报推导与贝尔曼期望图解。',
      actionType: 'lan-note',
      noteSource: 'RL-book',
      notePath: '马尔可夫奖励过程 / 回报'
    },
    {
      id: 'res-core-guide',
      type: '核心进阶笔记',
      title: '强化学习核心导引',
      desc: 'MRP/MDP/贝尔曼关系笔记，重点梳理期望方程与最优方程的分野。',
      actionType: 'lan-note',
      noteSource: '强化学习核心导引',
      notePath: '贝尔曼方程'
    },
    {
      id: 'res-deepseek-r1',
      type: '前沿研读支线',
      title: 'DeepSeek-R1',
      desc: '前沿论文研读支线，记录大规模强化学习与纯 RL 推理涌现思考。',
      actionType: 'lan-note',
      noteSource: 'DeepSeek-R1',
      notePath: '论文研读 / 强化学习支线'
    },
    {
      id: 'res-repository',
      type: '公开代码仓库',
      title: 'rl-learning 仓库',
      desc: '现有检查点与学习项目目录，包含随堂练习代码与实验骨架。',
      actionType: 'link',
      url: 'https://github.com/yonggodlikean/rl-learning',
      btnText: '访问 GitHub 仓库 ↗'
    }
  ];

  // =========================================================================
  // 2. 本地存储安全管理 (localStorage: rl_hub_state_v1)
  // =========================================================================

  let isLocalStorageAvailable = false;
  let memoryStorage = {
    reflections: {},
    reflectionTimestamps: {},
    lastVisited: null,
    activeTopicId: null
  };

  function checkStorageAvailability() {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        isLocalStorageAvailable = false;
        return;
      }
      // 不使用探针 key，仅通过读取自身 STORAGE_KEY 检验权限，不污染任何外部或 checkpoint 存储
      localStorage.getItem(STORAGE_KEY);
      isLocalStorageAvailable = true;
    } catch (e) {
      isLocalStorageAvailable = false;
      console.warn('localStorage 不可用，启用内存暂存模式:', e);
    }
  }

  checkStorageAvailability();

  function loadState() {
    if (!isLocalStorageAvailable) return memoryStorage;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return memoryStorage;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        if (!parsed.reflections || typeof parsed.reflections !== 'object') {
          parsed.reflections = {};
        }
        if (!parsed.reflectionTimestamps || typeof parsed.reflectionTimestamps !== 'object') {
          parsed.reflectionTimestamps = {};
        }
        // 严格保留所有顶层未知、孤立或扩展字段
        return parsed;
      }
    } catch (e) {
      console.warn('解析本地状态失败，保留内存副本:', e);
    }
    return memoryStorage;
  }

  function saveState(state) {
    // 始终先将状态同步至内存，保障降级与会话一致性
    memoryStorage = state;
    if (!isLocalStorageAvailable) {
      return false;
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      return true;
    } catch (e) {
      // 配额超限或写保护异常，立即标记 localStorage 不可用并报告写入失败
      isLocalStorageAvailable = false;
      console.warn('保存本地状态异常，切换为当前会话暂存模式:', e);
      return false;
    }
  }

  function getReflection(key) {
    const state = loadState();
    // 兼容对象形式 { text, savedAt } 与纯字符串形式
    if (state.reflections && state.reflections[key] != null) {
      const val = state.reflections[key];
      if (typeof val === 'object' && val.text != null) {
        return String(val.text);
      }
      if (typeof val === 'string') {
        return val;
      }
    }
    // 兼容历史直接保存在顶层的键
    if (state[key] != null) {
      const val = state[key];
      if (typeof val === 'object' && val.text != null) {
        return String(val.text);
      }
      if (typeof val === 'string') {
        return val;
      }
    }
    return '';
  }

  function getReflectionSavedTime(key) {
    const state = loadState();
    if (state.reflections && state.reflections[key] != null) {
      const val = state.reflections[key];
      if (typeof val === 'object' && val.savedAt != null) {
        return val.savedAt;
      }
    }
    // 兼容数值型时间戳备用映射 reflectionTimestamps
    if (state.reflectionTimestamps && state.reflectionTimestamps[key] != null) {
      return state.reflectionTimestamps[key];
    }
    // 兼容顶层历史键
    if (state[key] != null && typeof state[key] === 'object' && state[key].savedAt != null) {
      return state[key].savedAt;
    }
    return null;
  }

  function setReflection(key, text) {
    const state = loadState();
    state.reflections = state.reflections || {};
    state.reflectionTimestamps = state.reflectionTimestamps || {};

    const existingRecord = (typeof state.reflections[key] === 'object' && state.reflections[key] !== null)
      ? state.reflections[key]
      : {};

    const nowIso = new Date().toISOString();
    const nowTs = Date.now();

    // 仅更新该键，且保留其原有其他扩展字段；同时维护数值 fallback 时间戳
    state.reflections[key] = {
      ...existingRecord,
      text: text,
      savedAt: nowIso
    };
    state.reflectionTimestamps[key] = nowTs;

    return saveState(state);
  }

  function recordLastVisited(route, title) {
    if (!route || typeof route !== 'string') return null;
    const trimmedRoute = route.trim();
    if (!/^#\/(?:chapter|case|topic)\/(.+)$/.test(trimmedRoute)) {
      return null;
    }
    const state = loadState();
    state.lastVisited = {
      route: trimmedRoute,
      title: title,
      timestamp: Date.now()
    };
    return saveState(state);
  }

  function getLastVisited() {
    const state = loadState();
    return state.lastVisited || null;
  }

  // 防抖保存管理
  let pendingSaveTimeout = null;
  let pendingSaveAction = null;

  function scheduleSaveReflection(key, text, statusElement, immediate) {
    if (pendingSaveTimeout) {
      clearTimeout(pendingSaveTimeout);
      pendingSaveTimeout = null;
    }

    const doSave = () => {
      const saveOk = setReflection(key, text);
      const timeStr = new Date().toLocaleTimeString('zh-CN', { hour12: false });
      if (statusElement) {
        statusElement.textContent = (saveOk && isLocalStorageAvailable)
          ? `✔ 已保存在本地 (${timeStr})`
          : `⚠ 当前会话暂存 (${timeStr})`;
      }
      pendingSaveAction = null;
    };

    pendingSaveAction = doSave;

    if (immediate) {
      doSave();
    } else {
      if (statusElement) {
        statusElement.textContent = '… 正在保存';
      }
      pendingSaveTimeout = setTimeout(doSave, 400);
    }
  }

  function flushPendingReflectionSave() {
    if (pendingSaveTimeout) {
      clearTimeout(pendingSaveTimeout);
      pendingSaveTimeout = null;
    }
    if (typeof pendingSaveAction === 'function') {
      pendingSaveAction();
      pendingSaveAction = null;
    }
  }

  window.addEventListener('beforeunload', flushPendingReflectionSave);
  window.addEventListener('pagehide', flushPendingReflectionSave);

  // =========================================================================
  // 3. 安全转义与 URL 校验 (XSS & Injection Protection)
  // =========================================================================

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function sanitizeUrl(url) {
    if (!url || typeof url !== 'string') return '#';
    const trimmed = url.trim();
    if (!trimmed || trimmed.startsWith('javascript:') || trimmed.startsWith('data:') || trimmed.startsWith('vbscript:')) {
      return '#';
    }
    // 禁止任何协议相对 URL (如 //evil.com)
    if (trimmed.startsWith('//')) {
      return '#';
    }

    // 内部 hash 路由：仅允许合法字符，防范属性引号注入
    if (trimmed.startsWith('#')) {
      return trimmed.replace(/["'<>\s\0-\x1f]/g, c => encodeURIComponent(c));
    }

    // 相对路径 (以 / 或 ./ 开头，且非 //)
    if (trimmed.startsWith('/') || trimmed.startsWith('./')) {
      return trimmed.replace(/["'<>\s\0-\x1f]/g, c => encodeURIComponent(c));
    }

    // 绝对 HTTP / HTTPS URL
    if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
      try {
        const parsed = new URL(trimmed);
        if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
          return '#';
        }
        // 禁止包含认证信息 (userinfo，如 http://user:pass@domain)
        if (parsed.username || parsed.password) {
          return '#';
        }
        // http 协议仅允许本地回环与局域网网段
        if (parsed.protocol === 'http:') {
          const host = parsed.hostname;
          const isLoopbackOrLan =
            host === 'localhost' ||
            host === '127.0.0.1' ||
            host === '[::1]' ||
            host === '0.0.0.0' ||
            /^127\.\d+\.\d+\.\d+$/.test(host) ||
            /^10\.\d+\.\d+\.\d+$/.test(host) ||
            /^172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+$/.test(host) ||
            /^192\.168\.\d+\.\d+$/.test(host) ||
            host.endsWith('.local');
          if (!isLoopbackOrLan) {
            return '#';
          }
        }
        // 编码任何残留的双引号、单引号与尖括号，确保在未经过额外 escapeHtml 的 href 模板中绝对属性安全
        return parsed.href
          .replace(/"/g, '%22')
          .replace(/'/g, '%27')
          .replace(/</g, '%3C')
          .replace(/>/g, '%3E');
      } catch (e) {
        return '#';
      }
    }

    return '#';
  }

  // =========================================================================
  // 4. 数据目录快照与校验 (Catalog Fetch & Schema Validation)
  // =========================================================================

  let catalogData = null;
  let catalogLoading = true;
  let catalogError = null;

  async function fetchCatalog() {
    catalogLoading = true;
    catalogError = null;
    renderLoadingState();
    try {
      const response = await fetch(CATALOG_URL, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`无法获取 catalog.json (HTTP 状态码: ${response.status})`);
      }
      const data = await response.json();
      validateCatalog(data);
      catalogData = data;
      catalogLoading = false;
      onCatalogLoaded();
    } catch (err) {
      catalogLoading = false;
      catalogError = err.message || '加载 catalog.json 失败';
      renderErrorState(catalogError);
    }
  }

  function validateCatalog(data) {
    if (!data || typeof data !== 'object') {
      throw new Error('Catalog 根对象无效');
    }
    if (data.schema_version !== 1) {
      throw new Error(`不支持的 Catalog schema_version（当前只支持 1，接收到: ${data.schema_version}）`);
    }
    if (!Array.isArray(data.chapters)) {
      throw new Error('Catalog 结构缺失：chapters 必须为数组');
    }
    if (!Array.isArray(data.domains)) {
      throw new Error('Catalog 结构缺失：domains 必须为数组');
    }
    if (!Array.isArray(data.cases)) {
      throw new Error('Catalog 结构缺失：cases 必须为数组');
    }
  }

  // =========================================================================
  // 5. DOM 引用与基础交互 (Navigation, Dialog, Mobile Drawer)
  // =========================================================================

  const viewContainer = document.getElementById('viewContainer');
  const chapterNav = document.getElementById('chapterNav');
  const searchInput = document.getElementById('searchInput');
  const searchClearBtn = document.getElementById('searchClearBtn');
  const searchResultsDropdown = document.getElementById('searchResultsDropdown');
  const noteModal = document.getElementById('noteModal');
  const noteModalSource = document.getElementById('noteModalSource');
  const noteModalPath = document.getElementById('noteModalPath');
  const copyNotePathBtn = document.getElementById('copyNotePathBtn');
  const closeNoteModalBtn = document.getElementById('closeNoteModalBtn');
  const closeNoteModalIconBtn = document.getElementById('closeNoteModalIconBtn');
  const copyFeedback = document.getElementById('copyFeedback');
  const lanCardBtn = document.getElementById('lanCardBtn');
  const menuToggleBtn = document.getElementById('menuToggleBtn');
  const mobileSearchBtn = document.getElementById('mobileSearchBtn');
  const mobileBackdrop = document.getElementById('mobileBackdrop');
  const hubSidebar = document.getElementById('hubSidebar');
  const hubTopbar = document.querySelector('.hub-topbar');

  let lastActiveElementBeforeModal = null;
  let lastActiveElementBeforeSearch = null;

  // 移动端侧边栏抽屉无障碍状态同步
  function updateMobileDrawerState() {
    if (!hubSidebar) return;
    if (window.innerWidth <= 768) {
      const isOpen = hubSidebar.classList.contains('drawer-open');
      if (isOpen) {
        hubSidebar.removeAttribute('aria-hidden');
        hubSidebar.removeAttribute('inert');
      } else {
        hubSidebar.setAttribute('aria-hidden', 'true');
        hubSidebar.setAttribute('inert', '');
      }
    } else {
      hubSidebar.classList.remove('drawer-open');
      hubSidebar.removeAttribute('aria-hidden');
      hubSidebar.removeAttribute('inert');
      if (mobileBackdrop) {
        mobileBackdrop.classList.remove('open');
        mobileBackdrop.setAttribute('aria-hidden', 'true');
      }
      if (menuToggleBtn) {
        menuToggleBtn.setAttribute('aria-expanded', 'false');
      }
      if (hubTopbar) {
        hubTopbar.classList.remove('mobile-visible');
      }
    }
  }

  // 移动端侧边栏抽屉开关
  function openMobileDrawer() {
    if (!hubSidebar) return;
    hubSidebar.classList.add('drawer-open');
    hubSidebar.removeAttribute('aria-hidden');
    hubSidebar.removeAttribute('inert');
    if (mobileBackdrop) {
      mobileBackdrop.classList.add('open');
      mobileBackdrop.setAttribute('aria-hidden', 'false');
    }
    if (menuToggleBtn) {
      menuToggleBtn.setAttribute('aria-expanded', 'true');
    }
    // 移除 inert 并自动聚焦首个可用导航元素
    const firstFocusable = hubSidebar.querySelector('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])');
    if (firstFocusable && typeof firstFocusable.focus === 'function') {
      firstFocusable.focus();
    }
  }

  function closeMobileDrawer(restoreFocus = true) {
    if (!hubSidebar) return;
    hubSidebar.classList.remove('drawer-open');
    if (window.innerWidth <= 768) {
      hubSidebar.setAttribute('aria-hidden', 'true');
      hubSidebar.setAttribute('inert', '');
    } else {
      hubSidebar.removeAttribute('aria-hidden');
      hubSidebar.removeAttribute('inert');
    }
    if (mobileBackdrop) {
      mobileBackdrop.classList.remove('open');
      mobileBackdrop.setAttribute('aria-hidden', 'true');
    }
    if (menuToggleBtn) {
      menuToggleBtn.setAttribute('aria-expanded', 'false');
      if (restoreFocus && document.activeElement !== menuToggleBtn) {
        menuToggleBtn.focus();
      }
    }
  }

  // 移动端搜索栏切换控制
  function openMobileSearch() {
    lastActiveElementBeforeSearch = document.activeElement;
    if (hubTopbar) {
      hubTopbar.classList.add('mobile-visible');
    }
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }

  function closeMobileSearch(restoreFocus = true) {
    if (hubTopbar && hubTopbar.classList.contains('mobile-visible')) {
      hubTopbar.classList.remove('mobile-visible');
    }
    clearSearch();
    if (restoreFocus) {
      if (lastActiveElementBeforeSearch && typeof lastActiveElementBeforeSearch.focus === 'function') {
        lastActiveElementBeforeSearch.focus();
      } else if (mobileSearchBtn) {
        mobileSearchBtn.focus();
      }
    }
  }

  if (mobileSearchBtn) {
    mobileSearchBtn.addEventListener('click', () => {
      if (hubTopbar && hubTopbar.classList.contains('mobile-visible')) {
        closeMobileSearch(true);
      } else {
        openMobileSearch();
      }
    });
  }

  if (menuToggleBtn) {
    menuToggleBtn.addEventListener('click', () => {
      const isOpen = hubSidebar && hubSidebar.classList.contains('drawer-open');
      if (isOpen) {
        closeMobileDrawer(true);
      } else {
        openMobileDrawer();
      }
    });
  }

  if (mobileBackdrop) {
    mobileBackdrop.addEventListener('click', () => closeMobileDrawer(true));
  }

  // 点击抽屉内部链接时导航收起，不抢占焦点
  if (hubSidebar) {
    hubSidebar.addEventListener('click', (e) => {
      const link = e.target.closest('a');
      if (link && window.innerWidth <= 768) {
        closeMobileDrawer(false);
      }
    });
  }

  window.addEventListener('resize', updateMobileDrawerState);
  updateMobileDrawerState();

  // 局域网笔记 Native Dialog 控制
  function openNoteModal(sourceName, pathString) {
    lastActiveElementBeforeModal = document.activeElement;
    if (noteModalSource) noteModalSource.textContent = sourceName || '未指定笔记源';
    if (noteModalPath) noteModalPath.textContent = pathString || '未指定路径';
    if (copyFeedback) copyFeedback.textContent = '';
    if (noteModal && typeof noteModal.showModal === 'function') {
      noteModal.showModal();
    }
  }

  function closeNoteModal() {
    if (noteModal && typeof noteModal.close === 'function') {
      noteModal.close();
    }
    if (lastActiveElementBeforeModal && typeof lastActiveElementBeforeModal.focus === 'function') {
      lastActiveElementBeforeModal.focus();
    }
  }

  if (closeNoteModalBtn) closeNoteModalBtn.addEventListener('click', closeNoteModal);
  if (closeNoteModalIconBtn) closeNoteModalIconBtn.addEventListener('click', closeNoteModal);

  if (copyNotePathBtn) {
    copyNotePathBtn.addEventListener('click', async () => {
      const source = (noteModalSource && noteModalSource.textContent) || '';
      const path = (noteModalPath && noteModalPath.textContent) || '';
      const textToCopy = `笔记源: ${source}\n目标路径: ${path}\n私有根地址: ${NOTE_LAN_ROOT}`;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(textToCopy);
        } else {
          // Fallback
          const ta = document.createElement('textarea');
          ta.value = textToCopy;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        }
        if (copyFeedback) copyFeedback.textContent = '✔ 路径已复制到剪贴板';
      } catch (err) {
        if (copyFeedback) copyFeedback.textContent = '⚠ 复制失败，请手动选取文本';
      }
    });
  }

  if (lanCardBtn) {
    lanCardBtn.addEventListener('click', () => {
      openNoteModal('局域网知识库服务', '私有笔记主库检索');
    });
  }

  // =========================================================================
  // 6. 全局检索系统 (Global Search Index)
  // =========================================================================

  let searchIndex = [];

  function buildSearchIndex() {
    searchIndex = [];
    if (!catalogData) return;

    // 1. 索引章节
    if (Array.isArray(catalogData.chapters)) {
      catalogData.chapters.forEach(ch => {
        searchIndex.push({
          type: 'chapter',
          typeLabel: '课程章节',
          id: ch.id,
          title: ch.title,
          subtitle: `${ch.reference} · 共 ${ch.count} 题 · 满分 ${ch.total_score} 分`,
          searchable: `${ch.title} ${ch.id} ${ch.reference}`.toLowerCase(),
          url: `#/chapter/${encodeURIComponent(ch.id)}`
        });

        // 索引题目
        if (Array.isArray(ch.questions)) {
          ch.questions.forEach(q => {
            const typeMap = { choice: '概念选择', numeric: '手算推演', open: '原理解释', code: '代码演练' };
            const typeStr = typeMap[q.type] || q.type;
            searchIndex.push({
              type: 'question',
              typeLabel: `题目 (${typeStr})`,
              id: `${ch.id}/${q.id}`,
              title: q.title,
              subtitle: `所属章节: ${ch.title} · ${q.category} · ${q.max_score}分`,
              searchable: `${q.title} ${q.id} ${q.category} ${ch.title} ${ch.id} ${q.entry_point || ''}`.toLowerCase(),
              url: `#/chapter/${encodeURIComponent(ch.id)}`
            });
          });
        }
      });
    }

    // 2. 索引案例
    if (Array.isArray(catalogData.cases)) {
      catalogData.cases.forEach(c => {
        searchIndex.push({
          type: 'case',
          typeLabel: '实验案例',
          id: c.id,
          title: c.title,
          subtitle: `领域: ${c.domain_id} · 路径: ${c.path}`,
          searchable: `${c.title} ${c.id} ${c.domain_id} ${c.path} ${c.description || ''}`.toLowerCase(),
          url: `#/case/${encodeURIComponent(c.id)}`
        });
      });
    }

    // 3. 索引可选导读批注与公式
    KNOWLEDGE_TOPICS.forEach(topic => {
      searchIndex.push({
        type: 'topic',
        typeLabel: '重点导读',
        id: topic.id,
        title: topic.title,
        subtitle: `公式: ${topic.formulaText} · 导读自测`,
        searchable: `${topic.title} ${topic.formulaText} ${topic.conceptDiff} ${topic.intuition} ${topic.textbookRef}`.toLowerCase(),
        url: `#/topic/${encodeURIComponent(topic.id)}`
      });
    });

    // 4. 索引次级资源
    VERIFIED_RESOURCES.forEach(res => {
      searchIndex.push({
        type: 'resource',
        typeLabel: '学习资源',
        id: res.id,
        title: res.title,
        subtitle: `${res.type} · ${res.desc}`,
        searchable: `${res.title} ${res.type} ${res.desc} ${res.notePath || ''}`.toLowerCase(),
        url: `#/resources`
      });
    });
  }

  function handleSearchInput(query) {
    const trimmed = (query || '').trim().toLowerCase();
    if (!trimmed) {
      if (searchClearBtn) searchClearBtn.style.display = 'none';
      if (searchResultsDropdown) {
        searchResultsDropdown.style.display = 'none';
        searchResultsDropdown.innerHTML = '';
      }
      return;
    }

    if (searchClearBtn) searchClearBtn.style.display = 'flex';

    const results = searchIndex.filter(item => item.searchable.includes(trimmed)).slice(0, 12);

    renderSearchResults(results, trimmed);
  }

  function renderSearchResults(results, query) {
    if (!searchResultsDropdown) return;

    if (results.length === 0) {
      searchResultsDropdown.innerHTML = `
        <div class="search-empty-item">
          未检索到与 "<strong>${escapeHtml(query)}</strong>" 匹配的章节、题目、案例或推导
        </div>
      `;
      searchResultsDropdown.style.display = 'block';
      return;
    }

    // 按类型归类展示
    const grouped = {};
    results.forEach(r => {
      if (!grouped[r.typeLabel]) grouped[r.typeLabel] = [];
      grouped[r.typeLabel].push(r);
    });

    let html = '';
    Object.keys(grouped).forEach(label => {
      html += `<div class="search-group-header">${escapeHtml(label)}</div>`;
      grouped[label].forEach(item => {
        html += `
          <a href="${escapeHtml(item.url)}" class="search-result-item" data-url="${escapeHtml(item.url)}">
            <div class="search-result-row">
              <span class="search-result-title">${escapeHtml(item.title)}</span>
              <span class="search-result-badge">${escapeHtml(item.id)}</span>
            </div>
            <div class="search-result-desc">${escapeHtml(item.subtitle)}</div>
          </a>
        `;
      });
    });

    searchResultsDropdown.innerHTML = html;
    searchResultsDropdown.style.display = 'block';

    // 绑定点击事件，重置搜索并支持同路由重新触发
    searchResultsDropdown.querySelectorAll('.search-result-item').forEach(el => {
      el.addEventListener('click', (e) => {
        const targetUrl = el.getAttribute('data-url');
        clearSearch();
        closeMobileDrawer();
        if (targetUrl && location.hash === targetUrl) {
          handleRoute();
        }
      });
    });
  }

  function clearSearch() {
    if (searchInput) searchInput.value = '';
    if (searchClearBtn) searchClearBtn.style.display = 'none';
    if (searchResultsDropdown) {
      searchResultsDropdown.style.display = 'none';
      searchResultsDropdown.innerHTML = '';
    }
  }

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      handleSearchInput(e.target.value);
    });

    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (window.innerWidth <= 768 && hubTopbar && hubTopbar.classList.contains('mobile-visible')) {
          closeMobileSearch(true);
        } else {
          clearSearch();
          searchInput.blur();
        }
      }
    });
  }

  if (searchClearBtn) {
    searchClearBtn.addEventListener('click', () => {
      clearSearch();
      if (searchInput) searchInput.focus();
    });
  }

  // 全局快捷键与无障碍键盘行为管理
  document.addEventListener('keydown', (e) => {
    // '/' 聚焦搜索 (移动端同时展开搜索条)
    if (e.key === '/' && document.activeElement !== searchInput && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      if (window.innerWidth <= 768) {
        openMobileSearch();
      } else if (searchInput) {
        searchInput.focus();
        searchInput.select();
      }
      return;
    }

    // Escape 键层级关闭
    if (e.key === 'Escape') {
      // 若 native dialog 打开，跳过自定义逻辑，由原生 dialog 处理
      if (noteModal && (noteModal.open || noteModal.hasAttribute('open'))) {
        return;
      }
      if (hubTopbar && hubTopbar.classList.contains('mobile-visible')) {
        e.preventDefault();
        closeMobileSearch(true);
        return;
      }
      if (hubSidebar && hubSidebar.classList.contains('drawer-open')) {
        e.preventDefault();
        closeMobileDrawer(true);
        return;
      }
      if (searchResultsDropdown && searchResultsDropdown.style.display !== 'none') {
        clearSearch();
        return;
      }
    }

    // 抽屉打开时的键盘 Tab 焦点陷阱 (Tab trap)
    if (e.key === 'Tab') {
      // 原生 modal 打开时由其自行管理焦点，不执行侧边栏 trap
      if (noteModal && (noteModal.open || noteModal.hasAttribute('open'))) {
        return;
      }
      if (hubSidebar && hubSidebar.classList.contains('drawer-open')) {
        const focusables = Array.from(hubSidebar.querySelectorAll('a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'))
          .filter(el => el.offsetParent !== null || el.getClientRects().length > 0);
        if (focusables.length > 0) {
          const firstEl = focusables[0];
          const lastEl = focusables[focusables.length - 1];
          if (e.shiftKey && document.activeElement === firstEl) {
            e.preventDefault();
            lastEl.focus();
          } else if (!e.shiftKey && document.activeElement === lastEl) {
            e.preventDefault();
            firstEl.focus();
          }
        }
      }
    }
  });

  // 点击外部收起搜索结果；点击搜索结果中的链接导航时重置搜索并收起，不抢占焦点
  document.addEventListener('click', (e) => {
    if (searchResultsDropdown && !e.target.closest('.topbar-search-wrapper')) {
      searchResultsDropdown.style.display = 'none';
    }
    const searchLink = e.target.closest('#searchResultsDropdown a');
    if (searchLink) {
      closeMobileSearch(false);
      closeMobileDrawer(false);
    }
  });

  // =========================================================================
  // 7. 动态侧边栏渲染 (Dynamic Chapter Sidebar)
  // =========================================================================

  function renderSidebarChapters() {
    if (!chapterNav || !catalogData || !Array.isArray(catalogData.chapters)) return;

    let html = '';
    catalogData.chapters.forEach((ch, idx) => {
      const order = ch.order !== undefined ? ch.order : idx + 1;
      html += `
        <a href="#/chapter/${encodeURIComponent(ch.id)}" class="topic-nav-link" data-chapter-id="${escapeHtml(ch.id)}" id="chapterNav-${escapeHtml(ch.id)}">
          <span class="topic-index">${escapeHtml(order)}</span>
          <span class="chapter-nav-title catalog-break-word">${escapeHtml(ch.title)}</span>
        </a>
      `;
    });

    if (catalogData.chapters.length === 0) {
      html = `<div style="padding: 8px 10px; font-size: 12px; color: var(--ink-muted);">暂无注册章节</div>`;
    }

    chapterNav.innerHTML = html;
  }

  function updateActiveNavigation(activeRoute, activeId) {
    // 1. 全局导航高亮
    const navItems = document.querySelectorAll('.sidebar-nav .nav-item, .mobile-bottomnav .bottomnav-item');
    navItems.forEach(item => {
      const r = item.getAttribute('data-route');
      if (r === activeRoute) {
        item.classList.add('active');
        item.setAttribute('aria-current', 'page');
      } else {
        item.classList.remove('active');
        item.removeAttribute('aria-current');
      }
    });

    // 2. 章节侧边栏高亮
    const chapterLinks = document.querySelectorAll('#chapterNav .topic-nav-link');
    chapterLinks.forEach(link => {
      const chId = link.getAttribute('data-chapter-id');
      if (activeRoute === 'chapter' && chId === activeId) {
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
      } else {
        link.classList.remove('active');
        link.removeAttribute('aria-current');
      }
    });
  }

  // =========================================================================
  // 8. 路由与视图分发 (Hash Router & Views)
  // =========================================================================

  function handleRoute() {
    flushPendingReflectionSave();
    closeMobileDrawer();

    // 滚动回页面顶部
    window.scrollTo(0, 0);
    const mainContent = document.getElementById('mainContent');
    if (mainContent) mainContent.scrollTop = 0;

    const hash = location.hash || '#/';

    if (catalogLoading) {
      renderLoadingState();
      return;
    }

    if (catalogError) {
      renderErrorState(catalogError);
      return;
    }

    if (hash === '#/' || hash === '' || hash === '#') {
      updateActiveNavigation('home', null);
      renderHomeView();
    } else if (hash.startsWith('#/chapter/')) {
      const rawId = hash.slice('#/chapter/'.length);
      let chapterId = '';
      try {
        chapterId = decodeURIComponent(rawId);
      } catch (e) {
        chapterId = rawId;
      }
      updateActiveNavigation('chapter', chapterId);
      renderChapterView(chapterId);
    } else if (hash.startsWith('#/case/')) {
      const rawId = hash.slice('#/case/'.length);
      let caseId = '';
      try {
        caseId = decodeURIComponent(rawId);
      } catch (e) {
        caseId = rawId;
      }
      updateActiveNavigation('cases', caseId);
      renderCaseView(caseId);
    } else if (hash === '#/labs') {
      updateActiveNavigation('labs', null);
      renderLabsView();
    } else if (hash === '#/resources') {
      updateActiveNavigation('resources', null);
      renderResourcesView();
    } else if (hash.startsWith('#/topic/')) {
      const topicId = hash.slice('#/topic/'.length);
      updateActiveNavigation('topic', topicId);
      renderTopicDetailView(topicId);
    } else {
      updateActiveNavigation('', null);
      renderNotFoundView(hash);
    }
  }

  window.addEventListener('hashchange', handleRoute);

  function onCatalogLoaded() {
    buildSearchIndex();
    renderSidebarChapters();
    handleRoute();
  }

  // =========================================================================
  // 9. 视图渲染实现 (View Implementations)
  // =========================================================================

  // A. 加载中骨架屏
  function renderLoadingState() {
    if (!viewContainer) return;
    viewContainer.innerHTML = `
      <div class="catalog-overview-container">
        <div class="catalog-hero" style="animation: pulse 1.5s infinite ease-in-out;">
          <div class="catalog-hero-eyebrow">RL / LEARNING LAB</div>
          <div class="catalog-hero-title">正在加载强化学习目录中枢...</div>
          <p class="catalog-hero-desc">正在读取 catalog.json 快照，请稍候。</p>
        </div>
      </div>
    `;
  }

  // B. 异常容错卡片
  function renderErrorState(errorMessage) {
    if (!viewContainer) return;
    viewContainer.innerHTML = `
      <div class="catalog-overview-container">
        <div class="catalog-hero" style="border-left: 4px solid var(--accent-rust);">
          <div class="catalog-hero-eyebrow" style="color: var(--accent-rust);">数据加载异常</div>
          <div class="catalog-hero-title">无法读取或解析学习目录</div>
          <p class="catalog-hero-desc" style="color: var(--accent-rust);">
            ${escapeHtml(errorMessage)}
          </p>
          <div style="margin-top: 16px; padding: 14px; background: var(--bg-subtle); border-radius: 6px; font-size: 13px;">
            <p><strong>常见原因与恢复指引：</strong></p>
            <ol style="margin-left: 20px; margin-top: 8px; line-height: 1.6;">
              <li><strong>尚未生成快照：</strong>请在仓库根目录执行目录编译脚本：<br>
                <code class="catalog-command-code">python3 hub/build_catalog.py</code>
              </li>
              <li><strong>本地服务限制：</strong>浏览器安全策略阻止直接读取本地数据，需在仓库根目录启动仅限回环的安全 HTTP 服务并访问：<br>
                <code class="catalog-command-code">python3 -m http.server 8899 --bind 127.0.0.1 --directory hub</code><br>
                浏览器中打开：<a href="http://127.0.0.1:8899/" target="_blank" rel="noopener noreferrer">http://127.0.0.1:8899/</a>。
              </li>
            </ol>
          </div>
          <div style="margin-top: 20px;">
            <button type="button" id="retryFetchBtn" class="btn btn-primary">重新尝试加载</button>
          </div>
        </div>
      </div>
    `;

    const retryBtn = document.getElementById('retryFetchBtn');
    if (retryBtn) {
      retryBtn.addEventListener('click', fetchCatalog);
    }
  }

  // C. 未找到页面 (404)
  function renderNotFoundView(hash) {
    if (!viewContainer) return;
    viewContainer.innerHTML = `
      <div class="catalog-overview-container">
        <div class="catalog-hero">
          <div class="catalog-hero-eyebrow">404 ROUTE NOT FOUND</div>
          <div class="catalog-hero-title">未找到指定页面</div>
          <p class="catalog-hero-desc">目标路由 <code>${escapeHtml(hash)}</code> 不存在或已被调整。</p>
          <div style="margin-top: 20px;">
            <a href="#/" class="btn btn-primary">返回全景总览</a>
          </div>
        </div>
      </div>
    `;
  }

  // D. 总览页 (Overview View)
  function resolveValidLastVisit(lastVisit, catalogData) {
    if (!lastVisit || typeof lastVisit !== 'object') return null;
    if (!lastVisit.route || typeof lastVisit.route !== 'string') return null;
    const route = lastVisit.route.trim();

    const match = route.match(/^#\/(chapter|case|topic)\/(.+)$/);
    if (!match) return null;

    const type = match[1];
    const remainder = match[2];
    let rawId;
    try {
      rawId = decodeURIComponent(remainder);
    } catch (e) {
      return null;
    }
    if (!rawId || !rawId.trim()) return null;

    const chapters = (catalogData && Array.isArray(catalogData.chapters)) ? catalogData.chapters : [];
    const cases = (catalogData && Array.isArray(catalogData.cases)) ? catalogData.cases : [];

    if (type === 'chapter') {
      const matchedChapter = chapters.find(c => c && (c.id === rawId || c.id === remainder || String(c.id) === rawId));
      if (!matchedChapter || !matchedChapter.id) return null;
      const title = matchedChapter.title
        ? (matchedChapter.reference ? `${matchedChapter.title} (${matchedChapter.reference})` : matchedChapter.title)
        : String(matchedChapter.id);
      return {
        route: `#/chapter/${encodeURIComponent(String(matchedChapter.id))}`,
        title: title
      };
    }

    if (type === 'case') {
      const matchedCase = cases.find(c => c && (c.id === rawId || c.id === remainder || String(c.id) === rawId));
      if (!matchedCase || !matchedCase.id) return null;
      return {
        route: `#/case/${encodeURIComponent(String(matchedCase.id))}`,
        title: matchedCase.title || String(matchedCase.id)
      };
    }

    if (type === 'topic') {
      const topics = (typeof KNOWLEDGE_TOPICS !== 'undefined' && Array.isArray(KNOWLEDGE_TOPICS))
        ? KNOWLEDGE_TOPICS
        : (typeof KNOWLEDGE_TOPICS !== 'undefined' ? KNOWLEDGE_TOPICS : []);
      const matchedTopic = topics.find(t => t && (t.id === rawId || t.id === remainder || String(t.id) === rawId));
      if (!matchedTopic || !matchedTopic.id) return null;

      if (!matchedTopic.chapterId) return null;
      const chapterExists = chapters.some(c => c && (c.id === matchedTopic.chapterId || String(c.id) === String(matchedTopic.chapterId)));
      if (!chapterExists) return null;

      return {
        route: `#/topic/${encodeURIComponent(String(matchedTopic.id))}`,
        title: matchedTopic.title || String(matchedTopic.id)
      };
    }

    return null;
  }

  function renderHomeView() {
    if (!viewContainer || !catalogData) return;

    recordLastVisited('#/', '全景总览');

    const stats = catalogData.stats || {};
    const chapters = catalogData.chapters || [];
    const domains = catalogData.domains || [];
    const lastVisit = getLastVisited();
    const resolvedLastVisit = resolveValidLastVisit(lastVisit, catalogData);

    // 锚点卡片计算：真实校验过的最近访问 vs 推荐第一章开始
    let anchorCardHtml = '';
    if (resolvedLastVisit) {
      anchorCardHtml = `
        <div class="catalog-continue-card">
          <div class="catalog-continue-info">
            <span class="catalog-continue-tag">继续学习 (上次访问记录)</span>
            <span class="catalog-continue-title">${escapeHtml(resolvedLastVisit.title || '学习节点')}</span>
          </div>
          <a href="${escapeHtml(resolvedLastVisit.route)}" class="btn btn-primary">继续进入 ↗</a>
        </div>
      `;
    } else if (chapters.length > 0) {
      const firstChapter = chapters[0];
      anchorCardHtml = `
        <div class="catalog-continue-card" style="background-color: var(--bg-surface); border-color: var(--border-fine); border-left-color: var(--accent-pine);">
          <div class="catalog-continue-info">
            <span class="catalog-continue-tag" style="color: var(--accent-pine);">从头开始 (首章推荐)</span>
            <span class="catalog-continue-title">${escapeHtml(firstChapter.title)} (${escapeHtml(firstChapter.reference)})</span>
          </div>
          <a href="#/chapter/${encodeURIComponent(firstChapter.id)}" class="btn btn-primary">开始学习 ↗</a>
        </div>
      `;
    }

    // 章节卡片列表
    let chapterCardsHtml = '';
    chapters.forEach((ch, idx) => {
      const order = ch.order !== undefined ? ch.order : idx + 1;
      const matchingAnnotations = KNOWLEDGE_TOPICS.filter(t => t.chapterId === ch.id);
      let annotationPills = '';
      if (matchingAnnotations.length > 0) {
        annotationPills = `
          <div class="catalog-annotation-badge">
            含重点推导: ${matchingAnnotations.map(a => escapeHtml(a.title)).join('、')}
          </div>
        `;
      }

      chapterCardsHtml += `
        <a href="#/chapter/${encodeURIComponent(ch.id)}" class="catalog-chapter-card">
          <div class="catalog-chapter-card-top">
            <span class="catalog-order-badge">学习单元 ${escapeHtml(order)}</span>
            <span class="catalog-ref-pill">${escapeHtml(ch.reference)}</span>
          </div>
          <h3 class="catalog-chapter-title">${escapeHtml(ch.title)}</h3>
          ${annotationPills}
          <div class="catalog-chapter-meta-row">
            <span class="catalog-meta-tag">📝 ${escapeHtml(ch.count)} 题自检</span>
            <span class="catalog-meta-tag">💻 ${escapeHtml(ch.code_count || 0)} 项演练</span>
            <span class="catalog-meta-tag">🎯 满分 ${escapeHtml(ch.total_score)}</span>
          </div>
        </a>
      `;
    });

    if (chapters.length === 0) {
      chapterCardsHtml = `<div class="chapter-empty-box">题库中尚无注册章节，请在 checkpoint/chapters.py 中配置并执行构建。</div>`;
    }

    // 领域卡片列表
    let domainCardsHtml = '';
    domains.forEach(d => {
      const count = d.case_count || 0;
      const badgeClass = count > 0 ? 'has-cases' : 'empty';
      const countLabel = count > 0 ? `${count} 个案例` : '待收录';
      domainCardsHtml += `
        <a href="#/labs" class="catalog-domain-card">
          <div class="catalog-domain-header">
            <span class="catalog-domain-title">${escapeHtml(d.title)}</span>
            <span class="catalog-domain-badge ${badgeClass}">${escapeHtml(countLabel)}</span>
          </div>
          <span class="catalog-domain-path">${escapeHtml(d.path)}/</span>
        </a>
      `;
    });

    viewContainer.innerHTML = `
      <div class="catalog-overview-container">
        <!-- 纸墨克制 Hero -->
        <header class="catalog-hero">
          <div class="catalog-hero-eyebrow">RL / LEARNING LAB · 循序渐进</div>
          <h1 class="catalog-hero-title">学习目录，随仓库一起生长</h1>
          <p class="catalog-hero-desc">
            从马尔可夫决策过程到深度强化学习，全站内容严格依据题库注册表与实验源码物理结构动态生成。拒绝虚假掌握进度，以“读、验、写、例、思”闭环支撑本地自学。
          </p>
        </header>

        <!-- 数据统计横幅 -->
        <div class="catalog-stats-grid">
          <div class="catalog-stat-card">
            <span class="catalog-stat-value">${escapeHtml(stats.chapter_count || chapters.length)}</span>
            <span class="catalog-stat-label">编排章节数 (Chapters)</span>
          </div>
          <div class="catalog-stat-card">
            <span class="catalog-stat-value">${escapeHtml(stats.question_count || 0)}</span>
            <span class="catalog-stat-label">随堂思考自检题 (Questions)</span>
          </div>
          <div class="catalog-stat-card">
            <span class="catalog-stat-value">${escapeHtml(stats.code_count || 0)}</span>
            <span class="catalog-stat-label">短函数代码演练 (Code Tasks)</span>
          </div>
          <div class="catalog-stat-card">
            <span class="catalog-stat-value">${escapeHtml(stats.case_count || 0)}</span>
            <span class="catalog-stat-label">物理实验案例 (Repository Cases)</span>
          </div>
        </div>

        <!-- 最近访问 / 学习锚点 -->
        ${anchorCardHtml}

        <!-- 课程章节目录网格 -->
        <section class="catalog-section" aria-labelledby="chapterSectionTitle">
          <div class="catalog-section-header">
            <h2 class="catalog-section-title" id="chapterSectionTitle">课程章节目录 (按编排顺序)</h2>
            <span class="catalog-section-meta">共 ${escapeHtml(chapters.length)} 章</span>
          </div>
          <div class="catalog-chapter-grid">
            ${chapterCardsHtml}
          </div>
        </section>

        <!-- 仓库实验物理领域网格 -->
        <section class="catalog-section" aria-labelledby="domainSectionTitle">
          <div class="catalog-section-header">
            <h2 class="catalog-section-title" id="domainSectionTitle">仓库实验领域索引 (Repository Domains)</h2>
            <span class="catalog-section-meta">${escapeHtml(domains.length)} 大物理根目录</span>
          </div>
          <div class="catalog-domain-grid">
            ${domainCardsHtml}
          </div>
        </section>

        <!-- 构建披露与更新命令 (折叠展示) -->
        <details class="catalog-disclosure">
          <summary>快照构建与数据披露说明 (Build Disclosure)</summary>
          <div class="catalog-disclosure-body">
            <p><strong>生成时间：</strong>${escapeHtml(catalogData.generated_at || '未知')}</p>
            <p><strong>仓库源分支：</strong><code>${escapeHtml(catalogData.repository && catalogData.repository.branch || 'main')}</code></p>
            <p><strong>目录刷新命令：</strong><code class="catalog-command-code">${escapeHtml(catalogData.update && catalogData.update.command || 'python3 hub/build_catalog.py')}</code></p>
            <p><strong>静态发布命令：</strong><code class="catalog-command-code">${escapeHtml(catalogData.update && catalogData.update.publish_command || 'python3 checkpoint/publish_pages.py')}</code></p>
            <p class="catalog-break-word"><strong>严正提示：</strong>本站所有目录与题目均为构建期生成的静态快照，本地修改或推送不会在前端实时轮询生效，需执行发布命令打包到 <code>docs/hub/</code>。仓库新增内容不代表个人掌握度。</p>
          </div>
        </details>
      </div>
    `;
  }

  // E. 通用章节页 (Generic Chapter View)
  function renderChapterView(chapterId) {
    if (!viewContainer || !catalogData) return;

    const chapters = catalogData.chapters || [];
    const chapterIndex = chapters.findIndex(c => c.id === chapterId);
    const chapter = chapters[chapterIndex];

    if (!chapter) {
      viewContainer.innerHTML = `
        <div class="chapter-view-wrapper">
          <div class="chapter-header-card">
            <div class="chapter-header-eyebrow">
              <span class="chapter-badge-order">未收录章节</span>
            </div>
            <h1 class="chapter-header-title">未找到章节 "${escapeHtml(chapterId)}"</h1>
            <p style="color: var(--ink-muted);">该章节未在 checkpoint/chapters.py 中注册，或尚未运行 build_catalog.py 重新生成目录快照。</p>
            <div style="margin-top: 16px;">
              <a href="#/" class="btn btn-primary">返回学习目录</a>
            </div>
          </div>
        </div>
      `;
      return;
    }

    recordLastVisited(`#/chapter/${encodeURIComponent(chapter.id)}`, chapter.title);

    // 题目分类归纳：选择、手算、解释、代码 (保留组内原有顺序)
    const questions = chapter.questions || [];
    const groups = {
      choice: { title: '概念选择题 (Concept)', list: [] },
      numeric: { title: '手算推演题 (Hand Calculation)', list: [] },
      open: { title: '原理解释题 (Explanation)', list: [] },
      code: { title: '代码实操题 (Code Practice)', list: [] }
    };

    questions.forEach(q => {
      if (groups[q.type]) {
        groups[q.type].list.push(q);
      } else {
        if (!groups.open) groups.open = { title: '其他自检题', list: [] };
        groups.open.list.push(q);
      }
    });

    // 关联可选导读批注
    const matchingAnnotations = KNOWLEDGE_TOPICS.filter(t => t.chapterId === chapter.id);

    // 关联真实案例
    const allCases = catalogData.cases || [];
    const associatedCases = allCases.filter(c => Array.isArray(c.chapter_ids) && c.chapter_ids.includes(chapter.id));

    // 翻页导航 (纯顺序排布，不设门槛)
    const prevChapter = chapterIndex > 0 ? chapters[chapterIndex - 1] : null;
    const nextChapter = chapterIndex < chapters.length - 1 ? chapters[chapterIndex + 1] : null;

    // 反思存储键
    const reflectionKey = `chapter:${chapter.id}`;
    const savedReflection = getReflection(reflectionKey);
    const savedTime = getReflectionSavedTime(reflectionKey);

    let initialStatusText = '尚未撰写本章反思';
    if (savedTime) {
      const timeStr = new Date(savedTime).toLocaleTimeString('zh-CN', { hour12: false });
      initialStatusText = isLocalStorageAvailable
        ? `✔ 已保存在本地 (${timeStr})`
        : `⚠ 当前会话暂存 (${timeStr})`;
    }

    // 渲染题目组 HTML
    let questionGroupsHtml = '';
    const groupKeys = ['choice', 'numeric', 'open', 'code'];
    groupKeys.forEach(gKey => {
      const g = groups[gKey];
      if (g.list.length === 0) return;
      questionGroupsHtml += `
        <div class="chapter-group-card">
          <div class="chapter-group-header">
            <span>${escapeHtml(g.title)}</span>
            <span>${escapeHtml(g.list.length)} 题</span>
          </div>
          <div class="chapter-question-list">
            ${g.list.map(q => {
              const qUrl = q.url || `${chapter.checkpoint_url}&question=${encodeURIComponent(q.id)}`;
              return `
                <div class="chapter-question-row">
                  <div class="chapter-q-info">
                    <span class="chapter-q-id">[${escapeHtml(chapter.id)} / ${escapeHtml(q.id)}]</span>
                    <span class="chapter-q-title">${escapeHtml(q.title)}</span>
                  </div>
                  <div class="chapter-q-meta">
                    <span class="chapter-q-score">${escapeHtml(q.max_score)} 分</span>
                    <a href="${sanitizeUrl(qUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary btn-sm">答题 ↗</a>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    });

    if (questions.length === 0) {
      questionGroupsHtml = `<div class="chapter-empty-box">本章题库中暂无可自检题目。</div>`;
    }

    // 渲染代码任务 HTML (从 catalog questions 中筛选 code 类型且有 entry_point 的题)
    const codeTasks = questions.filter(q => q.type === 'code' && q.entry_point);
    let codeTasksHtml = '';
    if (codeTasks.length > 0) {
      codeTasksHtml = `
        <div class="chapter-group-card">
          <div class="chapter-group-header">
            <span>浏览器环境短函数编写</span>
            <span>${escapeHtml(codeTasks.length)} 项</span>
          </div>
          <div class="chapter-question-list">
            ${codeTasks.map(ct => {
              const cUrl = ct.url || `${chapter.checkpoint_url}&question=${encodeURIComponent(ct.id)}`;
              return `
                <div class="chapter-question-row">
                  <div class="chapter-q-info">
                    <span class="chapter-q-id">[${escapeHtml(ct.id)}]</span>
                    <div>
                      <span class="chapter-q-title">入口函数: <code>${escapeHtml(ct.entry_point)}</code></span>
                      <div style="font-size: 12px; color: var(--ink-muted); margin-top: 2px;">${escapeHtml(ct.title)}</div>
                    </div>
                  </div>
                  <div class="chapter-q-meta">
                    <a href="${sanitizeUrl(cUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm">前往编写 ↗</a>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    } else {
      codeTasksHtml = `<div class="chapter-empty-box">本章无独立短函数编写任务，核心侧重概念辨析与手算递推。</div>`;
    }

    // 渲染关联真实案例
    let casesHtml = '';
    if (associatedCases.length > 0) {
      casesHtml = `
        <div class="catalog-domain-grid">
          ${associatedCases.map(c => `
            <a href="#/case/${encodeURIComponent(c.id)}" class="catalog-domain-card">
              <div class="catalog-domain-header">
                <span class="catalog-domain-title">${escapeHtml(c.title)}</span>
                <span class="catalog-domain-badge has-cases">${escapeHtml(c.domain_id)}</span>
              </div>
              <p style="font-size: 12px; color: var(--ink-secondary);">${escapeHtml(c.description || '暂无描述')}</p>
              <span class="catalog-domain-path">${escapeHtml(c.path)}</span>
            </a>
          `).join('')}
        </div>
      `;
    } else {
      casesHtml = `
        <div class="chapter-empty-box">
          暂无直接关联的物理实验案例。<br>
          可参考领域目录在仓库中添加包含 <code>lab.json</code> 的案例目录。
        </div>
      `;
    }

    // 渲染导读批注卡片
    let annotationSectionHtml = '';
    if (matchingAnnotations.length > 0) {
      annotationSectionHtml = `
        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 8px;">
          ${matchingAnnotations.map(anno => `
            <div class="chapter-annotation-card">
              <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">
                <span class="chapter-annotation-title">重点导读：${escapeHtml(anno.title)}</span>
                <a href="#/topic/${encodeURIComponent(anno.id)}" class="btn btn-secondary btn-sm">进入核心推导与静态推演 ↗</a>
              </div>
              <div class="chapter-annotation-formula">${anno.formulaHtml}</div>
              <p style="font-size: 13px; color: var(--ink-secondary); line-height: 1.5;">${escapeHtml(anno.intuition)}</p>
            </div>
          `).join('')}
        </div>
      `;
    }

    viewContainer.innerHTML = `
      <div class="chapter-view-wrapper">
        <!-- 导航面包屑 -->
        <div class="chapter-nav-bar">
          <a href="#/" class="btn btn-secondary btn-sm">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M10 13L5 8l5-5"/>
            </svg>
            <span>返回学习目录</span>
          </a>
          <div style="font-size: 12.5px; color: var(--ink-muted);">
            全景总览 / 章节 / ${escapeHtml(chapter.title)}
          </div>
        </div>

        <!-- 章节主卡片 -->
        <article class="chapter-header-card" aria-labelledby="chapterHeaderTitle">
          <div class="chapter-header-eyebrow">
            <span class="chapter-badge-order">学习单元 ${escapeHtml(chapter.order !== undefined ? chapter.order : chapterIndex + 1)}</span>
            <span style="font-size: 12.5px; color: var(--ink-muted);">${escapeHtml(chapter.reference)}</span>
          </div>

          <h1 class="chapter-header-title catalog-break-word" id="chapterHeaderTitle">${escapeHtml(chapter.title)}</h1>

          <div class="chapter-header-meta">
            <span class="catalog-meta-tag">共 ${escapeHtml(chapter.count)} 题</span>
            <span class="catalog-meta-tag">满分 ${escapeHtml(chapter.total_score)} 分</span>
            <span class="catalog-meta-tag">${escapeHtml(chapter.code_count || 0)} 个代码演练</span>
          </div>

          <div class="chapter-header-actions">
            <a href="${sanitizeUrl(chapter.checkpoint_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary">
              进入 Checkpoint 答题检验 ↗
            </a>
            <a href="${sanitizeUrl(chapter.source_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary">
              在 GitHub 查看题库源码 ↗
            </a>
          </div>
        </article>

        <!-- 学习流导引条 (读→验→写→例→思) -->
        <nav class="chapter-flow-nav" aria-label="章节学习流">
          <span class="chapter-flow-step active">1. 概念阅读 (Read)</span>
          <span class="chapter-flow-divider">→</span>
          <span class="chapter-flow-step active">2. 理论自检 (Check)</span>
          <span class="chapter-flow-divider">→</span>
          <span class="chapter-flow-step active">3. 代码演练 (Code)</span>
          <span class="chapter-flow-divider">→</span>
          <span class="chapter-flow-step active">4. 真实案例 (Case)</span>
          <span class="chapter-flow-divider">→</span>
          <span class="chapter-flow-step active">5. 本地沉淀 (Reflect)</span>
        </nav>

        <!-- 1. 概念阅读 (Read) -->
        <section class="chapter-section" aria-labelledby="readSectionTitle">
          <div class="chapter-section-header">
            <h2 class="chapter-section-title" id="readSectionTitle">
              <span class="chapter-step-badge">1. 读</span> 概念阅读与批注参考
            </h2>
          </div>
          <div style="background: var(--bg-surface); border: 1px solid var(--border-fine); border-radius: 6px; padding: 16px 20px; font-size: 13.5px; line-height: 1.6;">
            <p><strong>教材参考定位：</strong>${escapeHtml(chapter.reference)}</p>
            <p style="color: var(--ink-muted); margin-top: 4px;">建议对照教材相关小节阅读基础概念与公式推导，然后再进行随堂理论自检。</p>
          </div>
          ${annotationSectionHtml}
        </section>

        <!-- 2. 理论自检 (Check) -->
        <section class="chapter-section" aria-labelledby="checkSectionTitle">
          <div class="chapter-section-header">
            <h2 class="chapter-section-title" id="checkSectionTitle">
              <span class="chapter-step-badge">2. 验</span> 随堂自检题目列表
            </h2>
            <span style="font-size: 12px; color: var(--ink-muted);">共 ${escapeHtml(chapter.count)} 题</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 14px;">
            ${questionGroupsHtml}
          </div>
        </section>

        <!-- 3. 代码演练 (Code) -->
        <section class="chapter-section" aria-labelledby="codeSectionTitle">
          <div class="chapter-section-header">
            <h2 class="chapter-section-title" id="codeSectionTitle">
              <span class="chapter-step-badge">3. 写</span> 浏览器短函数编写
            </h2>
          </div>
          ${codeTasksHtml}
        </section>

        <!-- 4. 真实案例 (Case) -->
        <section class="chapter-section" aria-labelledby="caseSectionTitle">
          <div class="chapter-section-header">
            <h2 class="chapter-section-title" id="caseSectionTitle">
              <span class="chapter-step-badge">4. 例</span> 关联物理实验案例
            </h2>
          </div>
          ${casesHtml}
        </section>

        <!-- 5. 本地沉淀 (Reflect) -->
        <section class="chapter-section" aria-labelledby="reflectSectionTitle">
          <div class="chapter-section-header">
            <h2 class="chapter-section-title" id="reflectSectionTitle">
              <span class="chapter-step-badge">5. 思</span> 章节深度思考与反思 (纯本地持久化)
            </h2>
          </div>
          <div class="reflection-card">
            <div class="reflection-prompt-box">
              <p class="reflection-question">
                💡 本章学习反思：本章推导中哪一步让您觉得最精妙或最容易混淆？手算或代码编写中是否有边界条件需要警惕？
              </p>
            </div>
            <div class="reflection-textarea-container">
              <label for="chapterReflectionInput" class="sr-only">本章反思笔记输入区</label>
              <textarea id="chapterReflectionInput" class="reflection-textarea" placeholder="在此记录本章核心心得或困惑（内容仅保存在当前浏览器本地，支持防抖自动保存与 Cmd/Ctrl + Enter 快捷键）..."></textarea>
              <div class="reflection-footer">
                <div class="reflection-status-group">
                  <span id="chapterSaveStatusText" class="save-status-text">${escapeHtml(initialStatusText)}</span>
                  <span id="chapterCharCount" class="char-counter">0 字</span>
                </div>
                <div class="reflection-btn-group">
                  <span class="keyboard-hint">Cmd/Ctrl + Enter 保存</span>
                  <button type="button" id="chapterManualSaveBtn" class="save-btn">保存本章心得</button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- 纯顺序翻页导航 -->
        <div class="chapter-pagination">
          <div>
            ${prevChapter ? `
              <a href="#/chapter/${encodeURIComponent(prevChapter.id)}" class="btn btn-secondary">
                ← 上一章: ${escapeHtml(prevChapter.title)}
              </a>
            ` : `<span style="font-size: 12px; color: var(--ink-faint);">已是首章</span>`}
          </div>
          <div>
            ${nextChapter ? `
              <a href="#/chapter/${encodeURIComponent(nextChapter.id)}" class="btn btn-secondary">
                下一章: ${escapeHtml(nextChapter.title)} →
              </a>
            ` : `<span style="font-size: 12px; color: var(--ink-faint);">已是末章</span>`}
          </div>
        </div>
      </div>
    `;

    // 绑定反思逻辑
    const reflectionInput = document.getElementById('chapterReflectionInput');
    const saveStatusText = document.getElementById('chapterSaveStatusText');
    const charCount = document.getElementById('chapterCharCount');
    const manualSaveBtn = document.getElementById('chapterManualSaveBtn');

    if (reflectionInput) {
      reflectionInput.value = savedReflection;
      if (charCount) charCount.textContent = `${savedReflection.length} 字`;

      reflectionInput.addEventListener('input', (e) => {
        const text = e.target.value;
        if (charCount) charCount.textContent = `${text.length} 字`;
        scheduleSaveReflection(reflectionKey, text, saveStatusText, false);
      });

      reflectionInput.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
          e.preventDefault();
          scheduleSaveReflection(reflectionKey, reflectionInput.value, saveStatusText, true);
        }
      });
    }

    if (manualSaveBtn && reflectionInput) {
      manualSaveBtn.addEventListener('click', () => {
        scheduleSaveReflection(reflectionKey, reflectionInput.value, saveStatusText, true);
      });
    }
  }

  // F. 通用实验案例页 (Generic Case View)
  function renderCaseView(caseId) {
    if (!viewContainer || !catalogData) return;

    const allCases = catalogData.cases || [];
    const targetCase = allCases.find(c => c.id === caseId);

    if (!targetCase) {
      viewContainer.innerHTML = `
        <div class="case-view-wrapper">
          <div class="case-header-card">
            <h1 class="case-title">未找到案例 "${escapeHtml(caseId)}"</h1>
            <p style="color: var(--ink-muted);">该案例未在仓库目录中扫描发现，或对应清单文件已被移动。</p>
            <div style="margin-top: 16px;">
              <a href="#/labs" class="btn btn-primary">返回实验全景</a>
            </div>
          </div>
        </div>
      `;
      return;
    }

    recordLastVisited(`#/case/${encodeURIComponent(targetCase.id)}`, targetCase.title);

    const chapters = catalogData.chapters || [];
    const validChapters = (targetCase.chapter_ids || []).map(chId => {
      const found = chapters.find(c => c.id === chId);
      return { id: chId, title: found ? found.title : chId, exists: !!found };
    });

    const validPrereqs = (targetCase.prerequisites || []).map(chId => {
      const found = chapters.find(c => c.id === chId);
      return { id: chId, title: found ? found.title : chId, exists: !!found };
    });

    const reflectionKey = `case:${targetCase.id}`;
    const savedReflection = getReflection(reflectionKey);
    const savedTime = getReflectionSavedTime(reflectionKey);

    let initialStatusText = '尚未撰写案例笔记';
    if (savedTime) {
      const timeStr = new Date(savedTime).toLocaleTimeString('zh-CN', { hour12: false });
      initialStatusText = isLocalStorageAvailable
        ? `✔ 已保存在本地 (${timeStr})`
        : `⚠ 当前会话暂存 (${timeStr})`;
    }

    viewContainer.innerHTML = `
      <div class="case-view-wrapper">
        <div class="chapter-nav-bar">
          <a href="#/labs" class="btn btn-secondary btn-sm">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M10 13L5 8l5-5"/>
            </svg>
            <span>返回实验全景</span>
          </a>
          <div style="font-size: 12.5px; color: var(--ink-muted);">
            实验案例 / ${escapeHtml(targetCase.domain_id)} / ${escapeHtml(targetCase.title)}
          </div>
        </div>

        <article class="case-header-card" aria-labelledby="caseMainTitle">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="catalog-domain-badge has-cases">${escapeHtml(targetCase.domain_id)}</span>
            <span class="catalog-domain-badge empty">状态: ${escapeHtml(targetCase.status || 'source')}</span>
            <span class="catalog-domain-badge empty">发现方式: ${escapeHtml(targetCase.discovery || 'readme')}</span>
          </div>

          <h1 class="case-title catalog-break-word" id="caseMainTitle">${escapeHtml(targetCase.title)}</h1>

          <p style="font-size: 14.5px; color: var(--ink-secondary); line-height: 1.6;">
            ${escapeHtml(targetCase.description || '暂无描述说明。')}
          </p>

          <!-- 严谨提醒横幅：未经在线验证 -->
          <div class="case-warning-banner">
            <strong>⚠ 实验运行环境未在线验证 (Runtime NOT verified)</strong><br>
            本平台不在线执行任何实验代码或命令，亦不提供虚假的“运行通过”状态。下方入口与命令行参数均为仓库源码静态元数据，请在本地 Python 环境中配置依赖并手动执行。
          </div>

          <!-- 元数据表格 -->
          <div class="case-meta-grid">
            <div class="case-meta-item">
              <span class="case-meta-label">仓库相对路径</span>
              <span class="case-meta-value code-path">${escapeHtml(targetCase.path)}</span>
            </div>
            <div class="case-meta-item">
              <span class="case-meta-label">指定执行入口</span>
              <span class="case-meta-value code-path">${escapeHtml(targetCase.entrypoint || '无指定 (请参阅 README)')}</span>
            </div>
          </div>

          <!-- 推荐命令行 -->
          <div>
            <span style="font-size: 12px; font-weight: 600; color: var(--ink-muted);">本地推荐执行命令 (参考文本):</span>
            <pre class="case-command-block"><code>${escapeHtml(targetCase.command || `# 请在本地切换至目录:\ncd ${targetCase.path}`)}</code></pre>
          </div>

          <!-- 外部链接 -->
          <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px;">
            ${targetCase.source_url ? `
              <a href="${sanitizeUrl(targetCase.source_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary">
                在 GitHub 查看源码目录 ↗
              </a>
            ` : ''}
            ${targetCase.readme_url ? `
              <a href="${sanitizeUrl(targetCase.readme_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary">
                阅读 README.md 文档 ↗
              </a>
            ` : ''}
          </div>
        </article>

        <!-- 关联章节与前置依赖 -->
        <section class="chapter-section">
          <h2 class="chapter-section-title">关联章节与知识前置</h2>
          <div style="background: var(--bg-surface); border: 1px solid var(--border-fine); border-radius: 6px; padding: 18px 20px; display: flex; flex-direction: column; gap: 12px;">
            <div>
              <span style="font-size: 13px; font-weight: 600;">对应关联章节：</span>
              <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px;">
                ${validChapters.length > 0 ? validChapters.map(vc => vc.exists ? `
                  <a href="#/chapter/${encodeURIComponent(vc.id)}" class="btn btn-secondary btn-sm">${escapeHtml(vc.title)} ↗</a>
                ` : `<span class="catalog-domain-badge empty">${escapeHtml(vc.title)}</span>`).join('') : '<span style="font-size: 12px; color: var(--ink-muted);">未指定关联章节（收录于对应领域独立实验池）</span>'}
              </div>
            </div>

            <div>
              <span style="font-size: 13px; font-weight: 600;">推荐前置章节：</span>
              <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px;">
                ${validPrereqs.length > 0 ? validPrereqs.map(vp => vp.exists ? `
                  <a href="#/chapter/${encodeURIComponent(vp.id)}" class="btn btn-secondary btn-sm">${escapeHtml(vp.title)} ↗</a>
                ` : `<span class="catalog-domain-badge empty">${escapeHtml(vp.title)}</span>`).join('') : '<span style="font-size: 12px; color: var(--ink-muted);">无严格前置依赖</span>'}
              </div>
            </div>
          </div>
        </section>

        <!-- 案例本地反思 -->
        <section class="chapter-section">
          <h2 class="chapter-section-title">案例实验记录与思考 (纯本地持久化)</h2>
          <div class="reflection-card">
            <div class="reflection-prompt-box">
              <p class="reflection-question">
                💡 记录此实验在本地运行时的收敛曲线、调参现象、超参数设置或遇到的报错：
              </p>
            </div>
            <div class="reflection-textarea-container">
              <label for="caseReflectionInput" class="sr-only">案例心得输入区</label>
              <textarea id="caseReflectionInput" class="reflection-textarea" placeholder="在此记录实验日志与消融观察（仅保存在当前浏览器本地，支持防抖自动保存与 Cmd/Ctrl + Enter 快捷键）..."></textarea>
              <div class="reflection-footer">
                <div class="reflection-status-group">
                  <span id="caseSaveStatusText" class="save-status-text">${escapeHtml(initialStatusText)}</span>
                  <span id="caseCharCount" class="char-counter">0 字</span>
                </div>
                <div class="reflection-btn-group">
                  <span class="keyboard-hint">Cmd/Ctrl + Enter 保存</span>
                  <button type="button" id="caseManualSaveBtn" class="save-btn">保存实验记录</button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    `;

    // 绑定案例反思逻辑
    const reflectionInput = document.getElementById('caseReflectionInput');
    const saveStatusText = document.getElementById('caseSaveStatusText');
    const charCount = document.getElementById('caseCharCount');
    const manualSaveBtn = document.getElementById('caseManualSaveBtn');

    if (reflectionInput) {
      reflectionInput.value = savedReflection;
      if (charCount) charCount.textContent = `${savedReflection.length} 字`;

      reflectionInput.addEventListener('input', (e) => {
        const text = e.target.value;
        if (charCount) charCount.textContent = `${text.length} 字`;
        scheduleSaveReflection(reflectionKey, text, saveStatusText, false);
      });

      reflectionInput.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
          e.preventDefault();
          scheduleSaveReflection(reflectionKey, reflectionInput.value, saveStatusText, true);
        }
      });
    }

    if (manualSaveBtn && reflectionInput) {
      manualSaveBtn.addEventListener('click', () => {
        scheduleSaveReflection(reflectionKey, reflectionInput.value, saveStatusText, true);
      });
    }
  }

  // G. 代码演练与实验全景 (Labs View)
  function renderLabsView() {
    if (!viewContainer || !catalogData) return;

    recordLastVisited('#/labs', '代码演练 & 实验全景');

    const chapters = catalogData.chapters || [];
    const domains = catalogData.domains || [];
    const allCases = catalogData.cases || [];

    // 1. 真实浏览器代码演练列表 (按章节组织)
    let browserTasksHtml = '';
    chapters.forEach(ch => {
      const codeQuestions = (ch.questions || []).filter(q => q.type === 'code' && q.entry_point);
      if (codeQuestions.length === 0) return;

      browserTasksHtml += `
        <div class="chapter-group-card" style="margin-bottom: 14px;">
          <div class="chapter-group-header">
            <span>${escapeHtml(ch.title)} (${escapeHtml(ch.reference)})</span>
            <span>${escapeHtml(codeQuestions.length)} 个代码任务</span>
          </div>
          <div class="chapter-question-list">
            ${codeQuestions.map(cq => {
              const url = cq.url || `${ch.checkpoint_url}&question=${encodeURIComponent(cq.id)}`;
              return `
                <div class="chapter-question-row">
                  <div class="chapter-q-info">
                    <span class="chapter-q-id">[${escapeHtml(ch.id)} / ${escapeHtml(cq.id)}]</span>
                    <div>
                      <span class="chapter-q-title">函数：<code>${escapeHtml(cq.entry_point)}</code></span>
                      <div style="font-size: 12px; color: var(--ink-muted); margin-top: 2px;">${escapeHtml(cq.title)}</div>
                    </div>
                  </div>
                  <div class="chapter-q-meta">
                    <span class="chapter-q-score">${escapeHtml(cq.max_score)} 分</span>
                    <a href="${sanitizeUrl(url)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm">进入编写 ↗</a>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    });

    if (!browserTasksHtml) {
      browserTasksHtml = `<div class="chapter-empty-box">当前题库中暂无可用的浏览器短函数练习。</div>`;
    }

    // 2. 7 大物理领域与案例
    let domainsHtml = '';
    domains.forEach(d => {
      const domainCases = allCases.filter(c => c.domain_id === d.id);
      let casesInDomainHtml = '';
      if (domainCases.length > 0) {
        casesInDomainHtml = `
          <div class="catalog-domain-grid" style="margin-top: 10px;">
            ${domainCases.map(dc => `
              <a href="#/case/${encodeURIComponent(dc.id)}" class="catalog-domain-card">
                <div class="catalog-domain-header">
                  <span class="catalog-domain-title">${escapeHtml(dc.title)}</span>
                  <span class="catalog-domain-badge has-cases">${escapeHtml(dc.discovery)}</span>
                </div>
                <p style="font-size: 12px; color: var(--ink-secondary); line-height: 1.4;">${escapeHtml(dc.description || '暂无描述')}</p>
                <span class="catalog-domain-path">${escapeHtml(dc.path)}</span>
              </a>
            `).join('')}
          </div>
        `;
      } else {
        casesInDomainHtml = `
          <div class="chapter-empty-box" style="margin-top: 10px; padding: 16px;">
            待收录：当前 <code>${escapeHtml(d.path)}/</code> 目录下尚无实验源码。可在该目录下放置实验脚本与 <code>lab.json</code> 进行关联。
          </div>
        `;
      }

      domainsHtml += `
        <div style="background: var(--bg-surface); border: 1px solid var(--border-fine); border-radius: 8px; padding: 18px 20px; margin-bottom: 16px;">
          <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div>
              <h3 style="font-size: 16px; font-weight: 700; color: var(--ink-primary);">${escapeHtml(d.title)}</h3>
              <span class="code-path" style="font-size: 11.5px;">${escapeHtml(d.path)}/</span>
            </div>
            ${d.source_url ? `
              <a href="${sanitizeUrl(d.source_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary btn-sm">在 GitHub 浏览目录 ↗</a>
            ` : ''}
          </div>
          ${casesInDomainHtml}
        </div>
      `;
    });

    viewContainer.innerHTML = `
      <div class="catalog-overview-container">
        <header class="catalog-hero">
          <div class="catalog-hero-eyebrow">LABS & EXPERIMENTS · 真实物理结构</div>
          <h1 class="catalog-hero-title">代码演练与物理实验全景</h1>
          <p class="catalog-hero-desc">
            区分“浏览器端随堂短函数编写”与“仓库本地物理实验”。严禁捏造未实现的假 Lab，如实呈现当前仓库代码骨架。
          </p>
        </header>

        <!-- 真实代码题 -->
        <section class="catalog-section">
          <div class="catalog-section-header">
            <h2 class="catalog-section-title">随堂浏览器代码演练 (由已注册章节提供)</h2>
          </div>
          ${browserTasksHtml}
        </section>

        <!-- 7 大物理领域目录 -->
        <section class="catalog-section">
          <div class="catalog-section-header">
            <h2 class="catalog-section-title">仓库 7 大实验领域物理索引 (Repository Domains)</h2>
          </div>
          ${domainsHtml}
        </section>
      </div>
    `;
  }

  // H. 资源与笔记页 (Resources View)
  function renderResourcesView() {
    if (!viewContainer) return;

    recordLastVisited('#/resources', '学习资源 & 笔记');

    let resourcesHtml = `
      <div class="catalog-chapter-grid">
        ${VERIFIED_RESOURCES.map(res => `
          <div class="catalog-chapter-card" style="cursor: default;">
            <div class="catalog-chapter-card-top">
              <span class="catalog-order-badge">${escapeHtml(res.type)}</span>
            </div>
            <h3 class="catalog-chapter-title">${escapeHtml(res.title)}</h3>
            <p style="font-size: 13px; color: var(--ink-secondary); line-height: 1.5;">${escapeHtml(res.desc)}</p>
            <div style="margin-top: auto; padding-top: 12px; border-top: 1px solid var(--border-fine);">
              ${res.actionType === 'lan-note' ? `
                <button type="button" class="btn btn-secondary btn-sm open-note-dialog-btn" data-source="${escapeHtml(res.noteSource)}" data-path="${escapeHtml(res.notePath)}">
                  查阅笔记路径指引 ↗
                </button>
              ` : `
                <a href="${sanitizeUrl(res.url)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm">
                  ${escapeHtml(res.btnText || '访问链接 ↗')}
                </a>
              `}
            </div>
          </div>
        `).join('')}
      </div>
    `;

    viewContainer.innerHTML = `
      <div class="catalog-overview-container">
        <header class="catalog-hero">
          <div class="catalog-hero-eyebrow">RESOURCES & NOTES · 知识外脑</div>
          <h1 class="catalog-hero-title">学习资源与笔记索引</h1>
          <p class="catalog-hero-desc">
            汇集随堂参考材料与局域网私有知识库索引。私有服务运行于个人局域网设备中，点击下方卡片可获取目标笔记的检索路径指引。
          </p>
        </header>

        <section class="catalog-section">
          ${resourcesHtml}
        </section>
      </div>
    `;

    bindNoteButtons();
  }

  function bindNoteButtons() {
    document.querySelectorAll('.open-note-dialog-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const source = btn.getAttribute('data-source');
        const path = btn.getAttribute('data-path');
        openNoteModal(source, path);
      });
    });
  }

  // I. 历史导读详情页 (Legacy Topic Detail View)
  function renderTopicDetailView(topicId) {
    if (!viewContainer) return;

    const topic = KNOWLEDGE_TOPICS.find(t => t.id === topicId);
    if (!topic) {
      renderNotFoundView(`#/topic/${topicId}`);
      return;
    }

    recordLastVisited(`#/topic/${encodeURIComponent(topic.id)}`, topic.title);

    // 动态从 catalogData 匹配章节标题与元数据
    let chapterTitle = topic.defaultChapterTitle;
    let chapterMeta = topic.textbookRef;
    let checkpointUrl = `https://yonggodlikean.github.io/rl-learning/checkpoint/?chapter=${encodeURIComponent(topic.chapterId)}`;

    if (catalogData && Array.isArray(catalogData.chapters)) {
      const ch = catalogData.chapters.find(c => c.id === topic.chapterId);
      if (ch) {
        chapterTitle = ch.title;
        chapterMeta = `${ch.count} 题 · 满分 ${ch.total_score} 分 · ${ch.reference}`;
        checkpointUrl = ch.checkpoint_url || checkpointUrl;
      }
    }

    function makeCheckpointUrl(chapterId, questionId) {
      return `${checkpointUrl}&question=${encodeURIComponent(questionId)}`;
    }

    const savedText = getReflection(topic.id);
    const savedTime = getReflectionSavedTime(topic.id);

    let initialStatusText = '尚未撰写反思记录';
    if (savedTime) {
      const timeStr = new Date(savedTime).toLocaleTimeString('zh-CN', { hour12: false });
      initialStatusText = isLocalStorageAvailable
        ? `✔ 已保存在本地 (${timeStr})`
        : `⚠ 当前会话暂存 (${timeStr})`;
    }

    viewContainer.innerHTML = `
      <div class="topic-detail-wrapper">
        <div class="detail-nav-bar">
          <a href="#/chapter/${encodeURIComponent(topic.chapterId)}" class="back-btn" id="detailBackBtn">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M10 13L5 8l5-5"/>
            </svg>
            <span>返回所属章节 (${escapeHtml(chapterTitle)})</span>
          </a>
          <div class="detail-breadcrumbs">
            ${escapeHtml(chapterTitle)} / ${escapeHtml(topic.textbookRef)}
          </div>
        </div>

        <article class="detail-header-card" aria-labelledby="detailMainTitle">
          <div class="detail-eyebrow">
            <span class="hero-badge-focus">${escapeHtml(topic.chapterId.toUpperCase())} 重点导读推导</span>
            <span class="hero-chapter-meta">${escapeHtml(chapterMeta)}</span>
          </div>

          <h1 class="detail-title" id="detailMainTitle">${escapeHtml(topic.title)}</h1>

          <!-- 1. 概念澄清与防混淆对照 -->
          <div class="detail-section">
            <h2 class="detail-section-title">1. 概念澄清与防混淆对照</h2>
            <div class="concept-box">
              <p><strong>直觉界定：</strong>${escapeHtml(topic.conceptDiff)}</p>
              <p style="margin-top: 6px;"><strong>计算逻辑：</strong>${escapeHtml(topic.intuition)}</p>
            </div>
          </div>

          <!-- 2. 核心数学公式推演 -->
          <div class="detail-section">
            <h2 class="detail-section-title">2. 核心数学公式推演</h2>
            <div class="formula-stage-box">
              <div class="formula-primary">${topic.formulaHtml}</div>
              <div class="formula-explanation">${topic.formulaExpandedHtml}</div>
            </div>
          </div>

          <!-- 3. 强具象静态手算/数据流面板 -->
          <div class="detail-section">
            <h2 class="detail-section-title">3. 具象数值推演 (静态数据流示例)</h2>
            <div class="calc-panel">
              <div class="calc-panel-header">
                <span class="calc-panel-title">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                    <rect x="2" y="2" width="12" height="12" rx="2"/>
                    <path d="M5 8h6M8 5v6"/>
                  </svg>
                  ${escapeHtml(topic.staticCalc.title)}
                </span>
                <span class="calc-static-badge">${escapeHtml(topic.staticCalc.badge)}</span>
              </div>

              <div class="calc-flow-container">
                ${topic.staticCalc.rows.map(row => `
                  <div class="calc-row">
                    <span class="calc-label">${escapeHtml(row.label)}:</span>
                    <span class="calc-val ${row.highlight ? 'highlight' : ''}">${escapeHtml(row.val)}</span>
                  </div>
                `).join('')}
              </div>

              <div class="calc-nuance-box">
                <strong>学术严谨性说明：</strong>
                <span>${escapeHtml(topic.staticCalc.nuance)}</span>
              </div>
            </div>
          </div>

          <!-- 4. 实操联动面板 -->
          <div class="detail-section">
            <h2 class="detail-section-title">4. 实操联动导引</h2>
            <div class="actions-panel">
              <div class="action-tile">
                <div class="tile-head">
                  <span>📖 阅读笔记</span>
                </div>
                <div class="tile-desc">${escapeHtml(topic.noteSource)}: ${escapeHtml(topic.notePath)}</div>
                <button type="button" class="tile-btn btn-secondary open-note-dialog-btn" data-source="${escapeHtml(topic.noteSource)}" data-path="${escapeHtml(topic.notePath)}">
                  <span>查阅笔记路径指引</span>
                </button>
              </div>

              <div class="action-tile">
                <div class="tile-head">
                  <span>🎯 检查点检验</span>
                </div>
                <div class="tile-desc">对应题目: ${topic.checkpointQuestions.join(', ')} (满分100体系)</div>
                <div class="checkpoint-links-group" style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;">
                  ${topic.checkpointQuestions.map(q => `
                    <a href="${sanitizeUrl(makeCheckpointUrl(topic.chapterId, q))}" target="_blank" rel="noopener noreferrer" class="tile-btn btn-secondary" style="flex: 1 1 auto; min-width: 72px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center; text-align: center; text-decoration: none;">
                      <span>题目 ${escapeHtml(q)} ↗</span>
                    </a>
                  `).join('')}
                </div>
              </div>

              <div class="action-tile">
                <div class="tile-head">
                  <span>💻 短函数编写</span>
                </div>
                <div class="tile-desc">${escapeHtml(topic.codeExercise.funcName)} (浏览器环境)</div>
                <a href="${sanitizeUrl(makeCheckpointUrl(topic.chapterId, topic.codeExercise.q))}" target="_blank" rel="noopener noreferrer" class="tile-btn btn-primary">
                  <span>在检查点中编写 ↗</span>
                </a>
              </div>
            </div>
          </div>

          <!-- 5. 深度反思笔记 (纯本地存储) -->
          <div class="detail-section">
            <h2 class="detail-section-title">5. 深度反思思考题 (本地独立持久化)</h2>
            <div class="reflection-card">
              <div class="reflection-prompt-box">
                <p class="reflection-question">💡 反思思考题：${escapeHtml(topic.reflectionPrompt)}</p>
              </div>

              <div class="reflection-textarea-container">
                <label for="reflectionInput" class="sr-only">反思心得输入区</label>
                <textarea id="reflectionInput" class="reflection-textarea" placeholder="在此写下对本知识点的思考推导、疑惑或心得（内容仅保存在当前浏览器本地，支持防抖自动保存与 Cmd/Ctrl + Enter 快捷键）..."></textarea>

                <div class="reflection-footer">
                  <div class="reflection-status-group">
                    <span id="saveStatusText" class="save-status-text">${escapeHtml(initialStatusText)}</span>
                    <span id="charCount" class="char-counter">0 字</span>
                  </div>

                  <div class="reflection-btn-group">
                    <span class="keyboard-hint">Cmd/Ctrl + Enter 强制保存</span>
                    <button type="button" id="manualSaveBtn" class="save-btn">
                      <span>保存反思记录</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </article>
      </div>
    `;

    bindNoteButtons();

    // 绑定反思编辑区逻辑
    const reflectionInput = document.getElementById('reflectionInput');
    const saveStatusText = document.getElementById('saveStatusText');
    const charCount = document.getElementById('charCount');
    const manualSaveBtn = document.getElementById('manualSaveBtn');

    if (reflectionInput) {
      reflectionInput.value = savedText;
      if (charCount) charCount.textContent = `${savedText.length} 字`;

      reflectionInput.addEventListener('input', (e) => {
        const text = e.target.value;
        if (charCount) charCount.textContent = `${text.length} 字`;
        scheduleSaveReflection(topic.id, text, saveStatusText, false);
      });

      reflectionInput.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
          e.preventDefault();
          scheduleSaveReflection(topic.id, reflectionInput.value, saveStatusText, true);
        }
      });
    }

    if (manualSaveBtn && reflectionInput) {
      manualSaveBtn.addEventListener('click', () => {
        scheduleSaveReflection(topic.id, reflectionInput.value, saveStatusText, true);
      });
    }
  }

  // =========================================================================
  // 10. 应用初始化入口
  // =========================================================================

  fetchCatalog();

})();
