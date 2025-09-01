#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from tqdm import tqdm


class IbisDocCrawler:
    def __init__(self, base_url: str = "https://ibis-project.org", output_dir: str = "data/docs"):
        self.base_url = base_url
        self.docs_url = f"{base_url}/docs/"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.visited = set()
        self.docs_data = []
        
    def crawl(self, max_pages: Optional[int] = None) -> List[Dict]:
        print(f"Starting crawl of {self.docs_url}")
        urls_to_visit = [self.docs_url]
        pages_crawled = 0
        
        while urls_to_visit and (max_pages is None or pages_crawled < max_pages):
            url = urls_to_visit.pop(0)
            
            if url in self.visited:
                continue
                
            if not self._is_valid_doc_url(url):
                continue
                
            print(f"Crawling: {url}")
            page_data = self._crawl_page(url)
            
            if page_data:
                self.docs_data.append(page_data)
                self.visited.add(url)
                pages_crawled += 1
                
                for link in page_data.get('links', []):
                    if link not in self.visited and link not in urls_to_visit:
                        urls_to_visit.append(link)
                
                time.sleep(0.5)
        
        self._save_data()
        return self.docs_data
    
    def _is_valid_doc_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.netloc in ['ibis-project.org', ''] and
            '/docs/' in url and
            not url.endswith(('.pdf', '.png', '.jpg', '.svg'))
        )
    
    def _crawl_page(self, url: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if not main_content:
                return None
            
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Untitled"
            
            code_examples = []
            for code_block in main_content.find_all(['pre', 'code']):
                code_text = code_block.get_text(strip=True)
                if code_text:
                    lang_class = code_block.get('class', [])
                    language = 'python'
                    for cls in lang_class:
                        if 'language-' in str(cls):
                            language = str(cls).replace('language-', '')
                            break
                    
                    code_examples.append({
                        'code': code_text,
                        'language': language
                    })
            
            content_markdown = markdownify(str(main_content), heading_style="ATX")
            
            links = []
            for link in main_content.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if self._is_valid_doc_url(full_url):
                    links.append(full_url)
            
            page_type = self._determine_page_type(url, title_text, content_markdown)
            
            return {
                'url': url,
                'title': title_text,
                'type': page_type,
                'content_markdown': content_markdown,
                'code_examples': code_examples,
                'links': links,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
    
    def _determine_page_type(self, url: str, title: str, content: str) -> str:
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        if 'api' in url_lower or 'reference' in url_lower:
            return 'api_reference'
        elif 'tutorial' in url_lower or 'getting-started' in url_lower:
            return 'tutorial'
        elif 'how-to' in url_lower or 'guide' in url_lower:
            return 'guide'
        elif 'backend' in url_lower:
            return 'backend_specific'
        elif 'sql' in title_lower or 'sql' in url_lower:
            return 'sql_comparison'
        elif 'example' in title_lower or 'example' in content_lower[:500]:
            return 'example'
        else:
            return 'documentation'
    
    def _save_data(self):
        output_file = self.output_dir / 'ibis_docs.jsonl'
        with open(output_file, 'w') as f:
            for item in self.docs_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.docs_data)} pages to {output_file}")
        
        summary_file = self.output_dir / 'crawl_summary.json'
        summary = {
            'total_pages': len(self.docs_data),
            'page_types': {},
            'total_code_examples': sum(len(d['code_examples']) for d in self.docs_data)
        }
        
        for doc in self.docs_data:
            page_type = doc['type']
            summary['page_types'][page_type] = summary['page_types'].get(page_type, 0) + 1
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")


def extract_ibis_sql_pairs(docs_data: List[Dict]) -> List[Dict]:
    pairs = []
    
    for doc in docs_data:
        if doc['type'] in ['sql_comparison', 'guide', 'example']:
            code_blocks = doc['code_examples']
            
            i = 0
            while i < len(code_blocks) - 1:
                block1 = code_blocks[i]
                block2 = code_blocks[i + 1]
                
                if (block1['language'] == 'python' and 
                    'ibis' in block1['code'].lower() and
                    block2['language'] in ['sql', 'text']):
                    
                    pairs.append({
                        'ibis_code': block1['code'],
                        'sql_code': block2['code'],
                        'source_url': doc['url'],
                        'context': doc['title']
                    })
                    i += 2
                else:
                    i += 1
    
    return pairs


if __name__ == "__main__":
    crawler = IbisDocCrawler()
    
    print("Starting Ibis documentation crawl...")
    docs = crawler.crawl(max_pages=50)
    
    print(f"\nCrawled {len(docs)} pages")
    
    pairs = extract_ibis_sql_pairs(docs)
    if pairs:
        pairs_file = Path('data/docs/ibis_sql_pairs.jsonl')
        with open(pairs_file, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
        print(f"Extracted {len(pairs)} Ibis-SQL pairs to {pairs_file}")