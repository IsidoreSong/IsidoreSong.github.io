
// --- Language and Path Setup ---
const isChinesePage = window.location.pathname.includes('/zh/');
const currentLanguage = isChinesePage ? 'zh' : 'en';

// --- Global State ---
let originalContent = null;
let currentContentUrl = null; // Tracks the currently loaded sub-page URL

// --- DOM Elements ---
const contentContainer = document.getElementById('main-content');
const languageToggle = document.getElementById('language-toggle');

// --- Event Listener Setup ---
document.addEventListener('DOMContentLoaded', onDOMLoaded);
document.getElementById('home-link').addEventListener('click', handleHomeNavigation);
languageToggle.addEventListener('click', handleLanguageToggle);

// --- Functions ---

function onDOMLoaded() {
  originalContent = contentContainer.innerHTML;
  updateNavigationLinks();

  // Setup navigation for dynamic content links
  document.querySelectorAll('.nav-link, [data-page]').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const page = link.getAttribute('data-page');
      if (page) {
        navigateToState(page, currentLanguage);
      }
    });
  });

  if (window.location.hash && window.location.hash.length > 2) {
    const page = window.location.hash.substring(2);
    navigateToState(page, currentLanguage);
  }
}

function navigateToState(page, lang) {
  const contentMap = {
    cv: lang === 'zh' ? 'CV/中文简历.html' : 'CV/2026Fall_Youzhe_Song_CV_phd.html',
    sop: lang === 'zh' ? 'SOP/个人陈述.html' : 'SOP/SOP.html',
    'research-map': lang === 'zh' ? 'AcademicMap/researchMap_zh.html' : 'AcademicMap/researchMap.html'
  };

  const titleMap = {
    cv: lang === 'zh' ? '简历' : 'CV',
    sop: lang === 'zh' ? '个人陈述' : 'SOP',
    'research-map': lang === 'zh' ? '研究地图' : 'Research Map'
  };

  const targetUrl = contentMap[page];
  if (!targetUrl) return;

  currentContentUrl = targetUrl;
  history.pushState({ page, lang }, titleMap[page], `#/${page}`);
  loadContent(currentContentUrl);
}

function handleHomeNavigation(e) {
  e.preventDefault();
  const homePath = isChinesePage ? '/zh/' : '/';
  if (currentContentUrl) {
    history.pushState({ page: 'home', lang: currentLanguage }, 'Home', homePath);
    showOriginalContent();
  } else {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

function handleLanguageToggle(e) {
  e.preventDefault();
  if (languageToggle.classList.contains('disabled')) return;
  const targetPath = isChinesePage ? '/' : '/zh/';
  window.location.href = targetPath + window.location.hash;
}

function updateNavigationLinks() {
    const cvLink = document.getElementById('cv-link');
    if (!cvLink) return;

    if (isChinesePage) {
        cvLink.textContent = '简历';
        document.getElementById('sop-link').textContent = '个人陈述';
        document.getElementById('research-map-link').textContent = '研究地图';
        document.getElementById('home-link').innerHTML = '首页<span class="dropdown-arrow"></span>';
    } else {
        cvLink.textContent = 'CV';
        document.getElementById('sop-link').textContent = 'SOP';
        document.getElementById('research-map-link').textContent = 'Research Map';
        document.getElementById('home-link').innerHTML = 'Home<span class="dropdown-arrow"></span>';
    }
}

function getCorrectedAssetPath(originalPath, fragmentBasePath) {
    if (!originalPath || originalPath.startsWith('http') || originalPath.startsWith('/') || originalPath.startsWith('#') || originalPath.startsWith('mailto:')) {
        return originalPath;
    }
    const dirs = ['CV', 'SOP', 'AcademicMap', 'assets'];
    let pathFromRoot;
    if (dirs.some(dir => originalPath.startsWith(dir + '/'))) {
        pathFromRoot = originalPath;
    } else {
        pathFromRoot = fragmentBasePath ? `${fragmentBasePath}/${originalPath}` : originalPath;
    }
    return `/${pathFromRoot}`.replace(/\/\//g, '/');
}

function loadContent(url) {
    const fetchUrl = `/${url}`.replace(/\/\//g, '/');
    const fragmentBasePath = url.substring(0, url.lastIndexOf('/'));

    fetch(fetchUrl)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.text();
        })
        .then(html => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');

            document.querySelectorAll('[data-dynamic-asset]').forEach(el => el.remove());

            const headContent = doc.querySelector('head');
            if (headContent) {
                const loadPromises = [];

                headContent.querySelectorAll('link[rel="stylesheet"]').forEach(link => {
                    const p = new Promise(resolve => {
                        const newLink = link.cloneNode(true);
                        newLink.href = getCorrectedAssetPath(newLink.getAttribute('href'), fragmentBasePath);
                        newLink.dataset.dynamicAsset = 'true';
                        newLink.onload = resolve;
                        newLink.onerror = resolve;
                        document.head.appendChild(newLink);
                    });
                    loadPromises.push(p);
                });

                headContent.querySelectorAll('style').forEach(style => {
                    const newStyle = style.cloneNode(true);
                    newStyle.dataset.dynamicAsset = 'true';
                    document.head.appendChild(newStyle);
                });

                Promise.all(loadPromises).then(() => {
                    const bodyContent = doc.querySelector('body');
                    if (bodyContent) {
                        bodyContent.querySelectorAll('img[src], source[src]').forEach(el => {
                            el.setAttribute('src', getCorrectedAssetPath(el.getAttribute('src'), fragmentBasePath));
                        });
                        bodyContent.querySelectorAll('a[href]').forEach(el => {
                            el.setAttribute('href', getCorrectedAssetPath(el.getAttribute('href'), fragmentBasePath));
                        });

                        contentContainer.innerHTML = bodyContent.innerHTML;
                        executeScripts(doc, fragmentBasePath);
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error loading content:', error);
            contentContainer.innerHTML = '<p>Error loading content.</p>';
        });
}

function executeScripts(doc, fragmentBasePath) {
    const scripts = Array.from(doc.querySelectorAll('body script, head script'));

    function loadScript(index) {
        if (index >= scripts.length) {
            initializeModal();
            window.scrollTo(0, 0);
            return;
        }

        const script = scripts[index];
        const newScript = document.createElement('script');
        newScript.dataset.dynamicAsset = 'true';

        if (script.src) {
            newScript.src = getCorrectedAssetPath(script.getAttribute('src'), fragmentBasePath);
            newScript.onload = () => loadScript(index + 1);
            newScript.onerror = () => { console.error(`Failed to load script: ${newScript.src}`); loadScript(index + 1); };
            document.body.appendChild(newScript);
        } else {
            newScript.textContent = script.textContent;
            document.body.appendChild(newScript);
            loadScript(index + 1);
        }
    }
    loadScript(0);
}

function showOriginalContent() {
    if (originalContent) {
        contentContainer.innerHTML = originalContent;
        document.querySelectorAll('[data-dynamic-asset]').forEach(el => el.remove());
        initializeModal();
        window.scrollTo(0, 0);
    }
}

function initializeModal() {
    const modal = document.getElementById('myModal');
    if (!modal) return;

    const modalImg = document.getElementById("img01");
    const images = document.getElementsByClassName('publication-image');
    const closeSpan = document.getElementsByClassName("close")[0];

    for (let img of images) {
        img.onclick = function() {
            modal.style.display = "block";
            modalImg.src = this.src;
        }
    }

    if(closeSpan) {
        closeSpan.onclick = function() {
            modal.style.display = "none";
        }
    }

    modal.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
}
