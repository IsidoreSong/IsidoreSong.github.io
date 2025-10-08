// Language toggle state
let currentLanguage = document.documentElement.lang === 'zh-CN' ? 'zh' : 'en';

// Mapping between English and Chinese pages
const languageMap = {
  'index.html': 'zh/index.html',
  'zh/index.html': 'index.html',
  'CV/2026Fall_Youzhe_Song_CV_phd.html': 'CV/中文简历.html',
  'CV/中文简历.html': 'CV/2026Fall_Youzhe_Song_CV_phd.html',
  'SOP/SOP.html': 'SOP/个人陈述.html',
  'SOP/个人陈述.html': 'SOP/SOP.html',
  'AcademicMap/researchMap.html': 'AcademicMap/researchMapChinese.html',
  'AcademicMap/researchMapChinese.html': 'AcademicMap/researchMap.html'
};

// Handle browser back/forward buttons
window.addEventListener('popstate', function(event) {
  if (event.state) {
    if (event.state.page === 'home' && event.state.lang === 'en') {
      window.location.href = 'index.html';
    } else if (event.state.page === 'cv') {
      currentContentUrl = event.state.lang === 'en' ? 'CV/2026Fall_Youzhe_Song_CV_phd.html' : 'CV/中文简历.html';
      loadContent(currentContentUrl);
      // Disable language toggle for CV/SOP pages
      document.getElementById('language-toggle').classList.add('disabled');
      document.getElementById('language-toggle').style.color = '#ccc';
      document.getElementById('language-toggle').style.cursor = 'not-allowed';
    } else if (event.state.page === 'sop') {
      currentContentUrl = event.state.lang === 'en' ? 'SOP/SOP.html' : 'SOP/个人陈述.html';
      loadContent(currentContentUrl);
      // Disable language toggle for CV/SOP pages
      document.getElementById('language-toggle').classList.add('disabled');
      document.getElementById('language-toggle').style.color = '#ccc';
      document.getElementById('language-toggle').style.cursor = 'not-allowed';
    } else if (event.state.page === 'research-map') {
      currentContentUrl = event.state.lang === 'en' ? 'AcademicMap/researchMap.html' : 'AcademicMap/researchMapChinese.html';
      loadContent(currentContentUrl);
      // Enable language toggle for Research Map page
      document.getElementById('language-toggle').classList.remove('disabled');
      document.getElementById('language-toggle').style.color = '#333';
      document.getElementById('language-toggle').style.cursor = 'pointer';
    } else if (event.state.page === 'home') {
      // 根据语言状态跳转到对应语言的主页
      if (event.state.lang === 'zh') {
        window.location.href = 'zh/index.html';
      } else {
        window.location.href = 'index.html';
      }
    } else {
      // Home page
      currentContentUrl = null;
      showOriginalContent();
      // Enable language toggle for home page
      document.getElementById('language-toggle').classList.remove('disabled');
      document.getElementById('language-toggle').style.color = '#333';
      document.getElementById('language-toggle').style.cursor = 'pointer';
    }
    // 更新导航链接
    updateNavigationLinks();
  } else {
    // Home page
    currentContentUrl = null;
    showOriginalContent();
    // Enable language toggle for home page
    document.getElementById('language-toggle').classList.remove('disabled');
    document.getElementById('language-toggle').style.color = '#333';
    document.getElementById('language-toggle').style.cursor = 'pointer';
    // 更新导航链接
    updateNavigationLinks();
  }
});

// Get the modal
var modal = document.getElementById('myModal');

// Get the image and insert it inside the modal - use its "alt" text as a caption
var modalImg = document.getElementById("img01");
var images = document.getElementsByClassName('publication-image');
for (var i = 0; i < images.length; i++) {
  var img = images[i];
  img.onclick = function(){
    modal.style.display = "block";
    modalImg.src = this.src;
  }
}

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function() { 
  modal.style.display = "none";
}

// When the user clicks anywhere on the modal, close it
modal.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}

// 页面内容加载功能
let originalContent = null;
let currentContentUrl = null;

// 保存原始内容
document.addEventListener('DOMContentLoaded', function() {
  originalContent = document.getElementById('main-content').innerHTML;
  
  // Check if there's a hash in the URL on page load
  if (window.location.hash) {
    const hash = window.location.hash.substring(1); // Remove the #
    console.log('Hash detected in Chinese page:', hash); // 调试信息
    
    if (hash === '/cv') {
      currentContentUrl = 'CV/2026Fall_Youzhe_Song_CV_phd.html';
      loadContent(currentContentUrl);
      // Disable language toggle for CV/SOP pages
      document.getElementById('language-toggle').classList.add('disabled');
      document.getElementById('language-toggle').style.color = '#ccc';
      document.getElementById('language-toggle').style.cursor = 'not-allowed';
    } else if (hash === '/zh/cv') {
      currentContentUrl = 'CV/中文简历.html';
      loadContent(currentContentUrl);
      // Disable language toggle for CV/SOP pages
      document.getElementById('language-toggle').classList.add('disabled');
      document.getElementById('language-toggle').style.color = '#ccc';
      document.getElementById('language-toggle').style.cursor = 'not-allowed';
    } else if (hash === '/sop') {
      currentContentUrl = 'SOP/SOP.html';
      loadContent(currentContentUrl);
      // Disable language toggle for CV/SOP pages
      document.getElementById('language-toggle').classList.add('disabled');
      document.getElementById('language-toggle').style.color = '#ccc';
      document.getElementById('language-toggle').style.cursor = 'not-allowed';
    } else if (hash === '/zh/sop') {
      currentContentUrl = 'SOP/个人陈述.html';
      loadContent(currentContentUrl);
      // Disable language toggle for CV/SOP pages
      document.getElementById('language-toggle').classList.add('disabled');
      document.getElementById('language-toggle').style.color = '#ccc';
      document.getElementById('language-toggle').style.cursor = 'not-allowed';
    } else if (hash === '/research-map') {
      currentContentUrl = 'AcademicMap/researchMap.html';
      loadContent(currentContentUrl);
      // Enable language toggle for Research Map page
      document.getElementById('language-toggle').classList.remove('disabled');
      document.getElementById('language-toggle').style.color = '#333';
      document.getElementById('language-toggle').style.cursor = 'pointer';
    } else if (hash === '/zh/research-map') {
      currentContentUrl = 'AcademicMap/researchMapChinese.html';
      loadContent(currentContentUrl);
      // Enable language toggle for Research Map page
      document.getElementById('language-toggle').classList.remove('disabled');
      document.getElementById('language-toggle').style.color = '#333';
      document.getElementById('language-toggle').style.cursor = 'pointer';
    } else if (hash === '/home' || hash === '') {
      // 根据当前语言状态跳转到对应语言的主页
      if (currentLanguage === 'zh') {
        window.location.href = 'zh/index.html';
      } else {
        window.location.href = 'index.html';
      }
    }
  } else {
    // Home page - ensure language toggle is enabled
    document.getElementById('language-toggle').classList.remove('disabled');
    document.getElementById('language-toggle').style.color = '#333';
    document.getElementById('language-toggle').style.cursor = 'pointer';
  }
  
  // 更新导航链接
  updateNavigationLinks();
});

// 添加点击事件监听器到主页锚点链接
document.addEventListener('click', function(e) {
  // 检查点击的元素是否是锚点链接
  if (e.target.tagName === 'A' && e.target.getAttribute('href').startsWith('#')) {
    e.preventDefault();
    const targetId = e.target.getAttribute('href').substring(1);
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
      // 滚动到目标元素
      targetElement.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  }
});

// 加载CV内容
document.getElementById('cv-link').addEventListener('click', function(e) {
  e.preventDefault();
  // 根据当前语言状态加载对应语言的CV
  if (currentLanguage === 'zh') {
    currentContentUrl = 'CV/中文简历.html';
    history.pushState({page: 'cv', lang: 'zh'}, '简历', '#/zh/cv');
  } else {
    currentContentUrl = 'CV/2026Fall_Youzhe_Song_CV_phd.html';
    history.pushState({page: 'cv', lang: 'en'}, 'CV', '#/cv');
  }
  loadContent(currentContentUrl);
  // Disable language toggle for CV/SOP pages
  document.getElementById('language-toggle').classList.add('disabled');
  document.getElementById('language-toggle').style.color = '#ccc';
  document.getElementById('language-toggle').style.cursor = 'not-allowed';
});

// 加载SOP内容
document.getElementById('sop-link').addEventListener('click', function(e) {
  e.preventDefault();
  // 根据当前语言状态加载对应语言的SOP
  if (currentLanguage === 'zh') {
    currentContentUrl = 'SOP/个人陈述.html';
    history.pushState({page: 'sop', lang: 'zh'}, '个人陈述', '#/zh/sop');
  } else {
    currentContentUrl = 'SOP/SOP.html';
    history.pushState({page: 'sop', lang: 'en'}, 'SOP', '#/sop');
  }
  loadContent(currentContentUrl);
  // Disable language toggle for CV/SOP pages
  document.getElementById('language-toggle').classList.add('disabled');
  document.getElementById('language-toggle').style.color = '#ccc';
  document.getElementById('language-toggle').style.cursor = 'not-allowed';
});

// 加载Research Map内容
document.getElementById('research-map-link').addEventListener('click', function(e) {
  e.preventDefault();
  // 根据当前语言状态加载对应语言的Research Map
  if (currentLanguage === 'zh') {
    currentContentUrl = 'AcademicMap/researchMapChinese.html';
    history.pushState({page: 'research-map', lang: 'zh'}, '研究地图', '#/zh/research-map');
  } else {
    currentContentUrl = 'AcademicMap/researchMap.html';
    history.pushState({page: 'research-map', lang: 'en'}, 'Research Map', '#/research-map');
  }
  loadContent(currentContentUrl);
  // Enable language toggle when content is loaded
  document.getElementById('language-toggle').classList.remove('disabled');
  document.getElementById('language-toggle').style.color = '#333';
  document.getElementById('language-toggle').style.cursor = 'pointer';
});

// 返回主页
document.getElementById('home-link').addEventListener('click', function(e) {
  e.preventDefault();
  // 根据当前语言状态跳转到对应语言的主页
  if (currentLanguage === 'zh') {
    window.location.href = 'zh/index.html';
  } else {
    window.location.href = 'index.html';
  }
});

// 语言切换功能
document.getElementById('language-toggle').addEventListener('click', function(e) {
  e.preventDefault();
  
  // Check if the toggle is disabled
  if (this.classList.contains('disabled')) {
    return;
  }
  
  // 如果在主页，根据当前语言状态切换到对应语言的主页
  if (!currentContentUrl) {
    if (currentLanguage === 'zh') {
      window.location.href = 'index.html';
    } else {
      window.location.href = 'zh/index.html';
    }
    return;
  }
  
  // 如果当前显示的是某个页面内容，则切换语言版本
  const translatedUrl = languageMap[currentContentUrl];
  if (translatedUrl) {
    // 检查对应的翻译文件是否存在
    fetch(translatedUrl)
      .then(response => {
        if (response.ok) {
          currentContentUrl = translatedUrl;
          loadContent(translatedUrl);
          // 切换语言状态
          currentLanguage = currentLanguage === 'zh' ? 'en' : 'zh';
          // Update URL with language parameter
          const pageName = translatedUrl.includes('CV/') ? 'cv' : 
                          translatedUrl.includes('SOP/') ? 'sop' : 
                          translatedUrl.includes('AcademicMap/') ? 'research-map' : '';
          if (pageName) {
            const langPrefix = currentLanguage === 'en' ? '' : '/zh';
            history.pushState({page: pageName, lang: currentLanguage}, '', `#${langPrefix}/${pageName}`);
          }
          // 更新导航链接
          updateNavigationLinks();
        } else {
          // 如果没有对应的翻译文件，不执行任何操作
        }
      })
      .catch(error => {
        // 如果请求出错，不执行任何操作
      });
  }
});

function loadContent(url) {
  fetch(url)
    .then(response => response.text())
    .then(html => {
      // Create a temporary DOM element to parse the HTML
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');
      
      // Clean up previously added content-specific stylesheets
      const contentStyles = document.querySelectorAll('link[data-content-style]');
      contentStyles.forEach(link => link.remove());
      
      // Clean up previously added content-specific style tags
      const contentStyleTags = document.querySelectorAll('style[data-content-style]');
      contentStyleTags.forEach(style => style.remove());
      
      // Extract the head content to get CSS links and style tags
      const headContent = doc.querySelector('head');
      if (headContent) {
        // Look for CSS links in the head
        const cssLinks = headContent.querySelectorAll('link[rel="stylesheet"]');
        cssLinks.forEach(link => {
          const href = link.getAttribute('href');
          // Fix CSS path if needed
          if (href && !href.startsWith('http')) {
            // Add directory prefix based on the content type
            if (url.includes('CV/') && !href.startsWith('CV/')) {
              link.setAttribute('href', 'CV/' + href);
            } else if (url.includes('SOP/') && !href.startsWith('SOP/')) {
              link.setAttribute('href', 'SOP/' + href);
            } else if (url.includes('AcademicMap/') && !href.startsWith('AcademicMap/')) {
              link.setAttribute('href', 'AcademicMap/' + href);
            }
          }
          // Add the CSS link to the main document head with a marker
          const existingLink = document.querySelector(`link[href="${link.getAttribute('href')}"]`);
          if (!existingLink) {
            link.setAttribute('data-content-style', 'true');
            document.head.appendChild(link.cloneNode(true));
          }
        });
        
        // Look for style tags in the head
        const styleTags = headContent.querySelectorAll('style');
        styleTags.forEach(style => {
          // Add the style tag to the main document head with a marker
          style.setAttribute('data-content-style', 'true');
          document.head.appendChild(style.cloneNode(true));
        });
      }
      
      // Extract the body content
      const bodyContent = doc.querySelector('body');
      if (bodyContent) {
        // Get the inner content of the body without the body tag itself
        let content = bodyContent.innerHTML;
        
        // Fix resource paths based on content type
        if (url.includes('CV/') || url.includes('SOP/') || url.includes('AcademicMap/')) {
          let prefix = '';
          if (url.includes('CV/')) {
            prefix = 'CV/';
          } else if (url.includes('SOP/')) {
            prefix = 'SOP/';
          } else if (url.includes('AcademicMap/')) {
            prefix = 'AcademicMap/';
          }
          
          // Fix paths for both single and double quoted src attributes
          content = content.replace(/src='([^']*)'/g, (match, p1) => {
            // If the path doesn't already start with the prefix and doesn't contain a full URL
            if (!p1.startsWith(prefix) && !p1.startsWith('http')) {
              return `src='${prefix}${p1}'`;
            }
            return match;
          });
          
          content = content.replace(/src="([^"]*)"/g, (match, p1) => {
            // If the path doesn't already start with the prefix and doesn't contain a full URL
            if (!p1.startsWith(prefix) && !p1.startsWith('http')) {
              return `src="${prefix}${p1}"`;
            }
            return match;
          });
          
          // Fix paths for href attributes (for links to other resources)
          content = content.replace(/href='([^']*)'/g, (match, p1) => {
            // If the path doesn't already start with the prefix and doesn't contain a full URL
            if (!p1.startsWith(prefix) && !p1.startsWith('http') && !p1.startsWith('#') && !p1.startsWith('mailto:')) {
              return `href='${prefix}${p1}'`;
            }
            return match;
          });
          
          content = content.replace(/href="([^"]*)"/g, (match, p1) => {
            // If the path doesn't already start with the prefix and doesn't contain a full URL
            if (!p1.startsWith(prefix) && !p1.startsWith('http') && !p1.startsWith('#') && !p1.startsWith('mailto:')) {
              return `href="${prefix}${p1}"`;
            }
            return match;
          });
        }
        
        document.getElementById('main-content').innerHTML = content;
        
        // Rebind modal events (if there are images in the new content)
        initializeModal();
        
        // Scroll to top
        window.scrollTo(0, 0);
      }
    })
    .catch(error => {
      console.error('Error loading content:', error);
      document.getElementById('main-content').innerHTML = '<p>加载内容时出错，请稍后再试。</p>';
    });
}

function showOriginalContent() {
  if (originalContent) {
    document.getElementById('main-content').innerHTML = originalContent;
    initializeModal();
    window.scrollTo(0, 0);
    
    // Clean up content-specific stylesheets when returning to home
    const contentStyles = document.querySelectorAll('link[data-content-style]');
    contentStyles.forEach(link => link.remove());
    
    // Clean up content-specific style tags when returning to home
    const contentStyleTags = document.querySelectorAll('style[data-content-style]');
    contentStyleTags.forEach(style => style.remove());
  }
}

function initializeModal() {
  // 重新初始化模态框功能
  var modal = document.getElementById('myModal');
  if (!modal) return;
  
  var modalImg = document.getElementById("img01");
  var images = document.getElementsByClassName('publication-image');
  
  // 移除旧的事件监听器
  for (var i = 0; i < images.length; i++) {
    images[i].onclick = null;
  }
  
  // 添加新的事件监听器
  for (var i = 0; i < images.length; i++) {
    var img = images[i];
    img.onclick = function(){
      modal.style.display = "block";
      modalImg.src = this.src;
    }
  }
}

// 更新导航链接以反映当前语言状态
function updateNavigationLinks() {
  if (currentLanguage === 'zh') {
    document.getElementById('cv-link').textContent = '简历';
    document.getElementById('sop-link').textContent = '个人陈述';
    document.getElementById('research-map-link').textContent = '研究地图';
    document.getElementById('home-link').innerHTML = '首页<span class="dropdown-arrow"></span>';
  } else {
    document.getElementById('cv-link').textContent = 'CV';
    document.getElementById('sop-link').textContent = 'SOP';
    document.getElementById('research-map-link').textContent = 'Research Map';
    document.getElementById('home-link').innerHTML = 'Home<span class="dropdown-arrow"></span>';
  }
}

// 页面加载完成后更新导航链接
document.addEventListener('DOMContentLoaded', function() {
  updateNavigationLinks();
});