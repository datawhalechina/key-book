import { defineConfig } from 'vitepress'

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL
const base = process.env.BASE || (isVercel ? '/' : '/key-book/')

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '钥匙书 KeyBook',
  description: '《机器学习理论导引》伴读笔记：概念解释 · 证明补充 · 案例分享',

  head: [
    ['link', { rel: 'icon', href: `${base}images/preface.jpg`.replace('//', '/') }],
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }],
    ['meta', { name: 'keywords', content: '机器学习理论,钥匙书,KeyBook,PAC学习,VC维,泛化界,稳定性,一致性,收敛率,遗憾界,Datawhale' }],
    ['meta', { name: 'author', content: 'Datawhale' }]
  ],

  markdown: {
    math: true
  },

  themeConfig: {
    logo: '/images/preface.jpg',
    siteTitle: '钥匙书 KeyBook',

    nav: [
      { text: '首页', link: '/' },
      {
        text: '章节',
        items: [
          { text: '序言', link: '/catalog' },
          { text: '第1章 预备知识', link: '/chapter1' },
          { text: '第2章 可学性', link: '/chapter2' },
          { text: '第3章 复杂度', link: '/chapter3' },
          { text: '第4章 泛化界', link: '/chapter4' },
          { text: '第5章 稳定性', link: '/chapter5' },
          { text: '第6章 一致性', link: '/chapter6' },
          { text: '第7章 收敛率', link: '/chapter7' },
          { text: '第8章 遗憾界', link: '/chapter8' }
        ]
      },
      { text: '附录', link: '/appendix' },
      { text: '符号表', link: '/notation' },
      { text: '参考文献', link: '/reference' }
    ],

    sidebar: [
      {
        text: '开始阅读',
        items: [
          { text: '序言', link: '/catalog' },
          { text: '符号表', link: '/notation' }
        ]
      },
      {
        text: '正文章节',
        items: [
          { text: '第1章 预备知识', link: '/chapter1' },
          { text: '第2章 可学性', link: '/chapter2' },
          { text: '第3章 复杂度', link: '/chapter3' },
          { text: '第4章 泛化界', link: '/chapter4' },
          { text: '第5章 稳定性', link: '/chapter5' },
          { text: '第6章 一致性', link: '/chapter6' },
          { text: '第7章 收敛率', link: '/chapter7' },
          { text: '第8章 遗憾界', link: '/chapter8' }
        ]
      },
      {
        text: '附录与参考',
        items: [
          { text: '附录', link: '/appendix' },
          { text: '参考文献', link: '/reference' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/key-book' }
    ],

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '未找到相关结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    footer: {
      message: '基于 CC BY-NC-SA 4.0 许可协议',
      copyright: 'Copyright © Datawhale'
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
})
