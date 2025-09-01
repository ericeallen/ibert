"""
Crawler for Ibis documentation and examples.
Extracts documentation, code examples, and SQL equivalents from ibis-project.org
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm


class IbisDocCrawler:
    """Crawl and extract content from Ibis documentation."""
    
    def __init__(self, base_url: str = "https://ibis-project.org", output_dir: str = "data/raw"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.visited_urls = set()
        self.delay = 0.5  # Be respectful to the server
        
    def crawl(self, start_paths: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Crawl Ibis documentation starting from specified paths.
        
        Args:
            start_paths: List of starting paths relative to base_url
            
        Returns:
            Dictionary with crawled content organized by type
        """
        if start_paths is None:
            start_paths = [
                "/docs/",
                "/reference/",
                "/tutorials/",
                "/how-to/",
                "/concepts/",
            ]
        
        all_content = {
            "documentation": [],
            "code_examples": [],
            "sql_mappings": [],
            "api_reference": []
        }
        
        # Crawl each section
        for path in start_paths:
            url = urljoin(self.base_url, path)
            print(f"Crawling section: {url}")
            self._crawl_recursive(url, all_content, max_depth=3)
        
        # Save the collected data
        self._save_data(all_content)
        
        return all_content
    
    def _crawl_recursive(self, url: str, content_dict: Dict, depth: int = 0, max_depth: int = 3):
        """Recursively crawl documentation pages."""
        if depth > max_depth or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content from the page
            page_content = self._extract_page_content(soup, url)
            
            # Categorize and store content
            self._categorize_content(page_content, content_dict)
            
            # Find links to other documentation pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                if self._is_valid_doc_link(href):
                    next_url = urljoin(url, href)
                    if urlparse(next_url).netloc == urlparse(self.base_url).netloc:
                        self._crawl_recursive(next_url, content_dict, depth + 1, max_depth)
                        
        except Exception as e:
            print(f"Error crawling {url}: {e}")
    
    def _extract_page_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract relevant content from a documentation page."""
        content = {
            "url": url,
            "title": "",
            "text": "",
            "code_blocks": [],
            "sql_blocks": [],
            "tables": []
        }
        
        # Extract title
        title_elem = soup.find(['h1', 'title'])
        if title_elem:
            content["title"] = title_elem.get_text(strip=True)
        
        # Extract main content area
        main_content = soup.find(['main', 'article', 'div'], class_=['content', 'main', 'documentation'])
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Extract text content (convert to markdown for better structure)
            content["text"] = md(str(main_content), strip=['script', 'style'])
            
            # Extract code blocks
            for code_block in main_content.find_all(['pre', 'code']):
                code_text = code_block.get_text(strip=True)
                language = self._detect_language(code_block)
                
                if language == 'sql':
                    content["sql_blocks"].append(code_text)
                else:
                    content["code_blocks"].append({
                        "language": language,
                        "code": code_text
                    })
            
            # Extract tables (might contain API documentation)
            for table in main_content.find_all('table'):
                table_data = self._parse_table(table)
                if table_data:
                    content["tables"].append(table_data)
        
        return content
    
    def _detect_language(self, code_element) -> str:
        """Detect the programming language of a code block."""
        # Check for language class
        classes = code_element.get('class', [])
        for cls in classes:
            if 'language-' in cls:
                return cls.replace('language-', '')
            if cls in ['python', 'sql', 'bash', 'javascript']:
                return cls
        
        # Check parent for language hints
        parent = code_element.parent
        if parent:
            parent_classes = parent.get('class', [])
            for cls in parent_classes:
                if 'language-' in cls:
                    return cls.replace('language-', '')
        
        # Default to python for Ibis documentation
        return 'python'
    
    def _parse_table(self, table) -> Optional[Dict]:
        """Parse HTML table into structured data."""
        headers = []
        rows = []
        
        # Extract headers
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        elif table.find('tr'):
            # First row might be headers
            first_row = table.find('tr')
            headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
        
        # Extract data rows
        tbody = table.find('tbody') or table
        for row in tbody.find_all('tr')[1 if not header_row else 0:]:
            row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if row_data:
                rows.append(row_data)
        
        if headers or rows:
            return {"headers": headers, "rows": rows}
        return None
    
    def _is_valid_doc_link(self, href: str) -> bool:
        """Check if a link points to documentation content."""
        if not href or href.startswith('#') or href.startswith('mailto:'):
            return False
        
        # Include paths that likely contain documentation
        doc_patterns = ['/docs/', '/reference/', '/api/', '/tutorial/', '/guide/', '/how-to/', '/concepts/']
        
        # Exclude non-documentation links
        exclude_patterns = ['.pdf', '.zip', '.tar', '/github.com/', '/twitter.com/']
        
        href_lower = href.lower()
        
        if any(pattern in href_lower for pattern in exclude_patterns):
            return False
        
        return any(pattern in href_lower for pattern in doc_patterns)
    
    def _categorize_content(self, page_content: Dict, content_dict: Dict):
        """Categorize extracted content into appropriate buckets."""
        url = page_content['url']
        
        # Determine content type based on URL and content
        if '/api/' in url or '/reference/' in url:
            content_dict['api_reference'].append(page_content)
        elif any(page_content['sql_blocks']):
            content_dict['sql_mappings'].append(page_content)
        elif any(page_content['code_blocks']):
            content_dict['code_examples'].append(page_content)
        else:
            content_dict['documentation'].append(page_content)
    
    def _save_data(self, content_dict: Dict):
        """Save crawled data to JSON files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for content_type, content_list in content_dict.items():
            if content_list:
                output_file = self.output_dir / f"ibis_{content_type}_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(content_list, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(content_list)} items to {output_file}")


class IbisExampleExtractor:
    """Extract Ibis-to-SQL mappings and code examples."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def extract_mappings(self) -> List[Tuple[str, str]]:
        """
        Extract Ibis code to SQL mappings from documentation.
        
        Returns:
            List of (ibis_code, sql_code) tuples
        """
        mappings = []
        
        # Load SQL mapping files
        for json_file in self.data_dir.glob("ibis_sql_mappings_*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for page in data:
                ibis_codes = [block['code'] for block in page['code_blocks'] 
                             if block['language'] == 'python']
                sql_codes = page['sql_blocks']
                
                # Try to pair Ibis code with SQL
                for i, ibis_code in enumerate(ibis_codes):
                    if i < len(sql_codes):
                        mappings.append((ibis_code, sql_codes[i]))
        
        return mappings
    
    def extract_examples(self) -> List[Dict]:
        """
        Extract code examples with context.
        
        Returns:
            List of example dictionaries with code and description
        """
        examples = []
        
        for json_file in self.data_dir.glob("ibis_code_examples_*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for page in data:
                for code_block in page['code_blocks']:
                    if code_block['language'] == 'python':
                        example = {
                            'code': code_block['code'],
                            'context': page['title'],
                            'description': page['text'][:500],  # First 500 chars as description
                            'url': page['url']
                        }
                        examples.append(example)
        
        return examples


if __name__ == "__main__":
    # Example usage
    crawler = IbisDocCrawler()
    
    # Start with a smaller crawl for testing
    test_paths = ["/docs/", "/tutorials/"]  # Start with just docs and tutorials
    
    print("Starting Ibis documentation crawl...")
    content = crawler.crawl(test_paths)
    
    print("\nCrawl completed. Summary:")
    for content_type, items in content.items():
        print(f"  {content_type}: {len(items)} items")
    
    # Extract mappings
    extractor = IbisExampleExtractor()
    mappings = extractor.extract_mappings()
    examples = extractor.extract_examples()
    
    print(f"\nExtracted {len(mappings)} Ibis-to-SQL mappings")
    print(f"Extracted {len(examples)} code examples")