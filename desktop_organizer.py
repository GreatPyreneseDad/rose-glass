#!/usr/bin/env python3
"""
Desktop Deep Organizer
Organizes both files AND folders on your desktop for ultimate cleanliness!
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# File categories (same as before)
FILE_CATEGORIES = {
    'Images': ['.jpg', '.jpeg', '.png', '.gif', '.heic', '.bmp', '.svg', '.webp', '.ico', '.tiff'],
    'Documents': ['.pdf', '.doc', '.docx', '.txt', '.pages', '.odt', '.rtf', '.tex'],
    'Spreadsheets': ['.xls', '.xlsx', '.numbers', '.csv', '.ods'],
    'Presentations': ['.ppt', '.pptx', '.key', '.odp'],
    'Videos': ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm'],
    'Audio': ['.mp3', '.m4a', '.wav', '.aac', '.flac', '.aiff', '.ogg', '.wma'],
    'Archives': ['.zip', '.dmg', '.tar', '.gz', '.rar', '.7z', '.pkg', '.deb'],
    'Code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.sh', '.c', '.php', '.rb', '.go'],
    'Screenshots': ['.png', '.jpg'],  # Special handling for screenshots
}

# Folder categories - intelligently group folders
FOLDER_CATEGORIES = {
    'Projects': ['project', 'proj', 'work', 'dev', 'development', 'code', 'github'],
    'Work': ['work', 'job', 'office', 'business', 'client', 'company'],
    'School': ['school', 'class', 'course', 'homework', 'study', 'university', 'college'],
    'Personal': ['personal', 'private', 'me', 'my'],
    'Archives': ['old', 'archive', 'backup', 'past', 'previous', '2020', '2021', '2022', '2023'],
    'Temp': ['temp', 'tmp', 'test', 'testing', 'trash', 'delete'],
}

def detect_screenshot(file_path):
    """Check if file is a screenshot based on name pattern"""
    name = file_path.name.lower()
    screenshot_patterns = ['screenshot', 'screen shot', 'captur', 'grab']
    return any(pattern in name for pattern in screenshot_patterns)

def detect_folder_category(folder_name):
    """Intelligently categorize folders based on name"""
    folder_lower = folder_name.lower()
    
    for category, keywords in FOLDER_CATEGORIES.items():
        if any(keyword in folder_lower for keyword in keywords):
            return category
    
    return None  # Don't categorize if unsure

def get_file_age_category(file_path):
    """Categorize files by age"""
    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - mod_time
    
    if age < timedelta(days=7):
        return "Recent"
    elif age < timedelta(days=30):
        return "LastMonth"
    elif age < timedelta(days=365):
        return "ThisYear"
    else:
        return "Older"

def analyze_desktop():
    """Analyze desktop contents"""
    desktop = Path.home() / "Desktop"
    
    print("\n🔍 Analyzing your Desktop...")
    print("=" * 60)
    
    # Categorize everything
    files_by_type = {}
    folders_by_category = {}
    screenshots = []
    total_items = 0
    
    for item in desktop.iterdir():
        # Skip hidden files and our organization folders
        if item.name.startswith('.'):
            continue
            
        total_items += 1
        
        if item.is_file():
            # Check if screenshot
            if detect_screenshot(item):
                screenshots.append(item)
            else:
                # Regular file categorization
                ext = item.suffix.lower()
                category = "Other"
                for cat, extensions in FILE_CATEGORIES.items():
                    if ext in extensions:
                        category = cat
                        break
                
                if category not in files_by_type:
                    files_by_type[category] = []
                files_by_type[category].append(item)
                
        elif item.is_dir():
            # Skip if it's one of our organization folders
            if item.name in ['_FilesOrganized', '_FoldersOrganized', '_Screenshots', '_Recent']:
                continue
                
            category = detect_folder_category(item.name)
            if category:
                if category not in folders_by_category:
                    folders_by_category[category] = []
                folders_by_category[category].append(item)
    
    # Show analysis
    print(f"\n📊 Found {total_items} items on your Desktop:\n")
    
    if files_by_type:
        print("📄 FILES:")
        for cat, files in sorted(files_by_type.items()):
            print(f"  {cat}: {len(files)} files")
    
    if screenshots:
        print(f"  Screenshots: {len(screenshots)} files")
    
    if folders_by_category:
        print("\n📁 FOLDERS:")
        for cat, folders in sorted(folders_by_category.items()):
            print(f"  {cat}: {len(folders)} folders")
    
    return files_by_type, folders_by_category, screenshots

def show_organization_options():
    """Show different organization strategies"""
    print("\n🎯 How would you like to organize your Desktop?")
    print("=" * 60)
    print("1. Smart Organize - Files by type, folders by category")
    print("2. By Type Only - Just organize files by type")
    print("3. By Date - Organize by when items were last modified")
    print("4. Screenshots - Just organize screenshots")
    print("5. Deep Clean - Everything including subfolders")
    print("6. Custom - Choose what to organize")
    print("7. Cancel")
    
    return input("\n👉 Your choice (1-7): ").strip()

def smart_organize(files_by_type, folders_by_category, screenshots):
    """Smart organization of desktop"""
    desktop = Path.home() / "Desktop"
    moved_count = 0
    
    print("\n🚀 Smart Organizing your Desktop...")
    
    # Create main organization folders
    files_organized = desktop / "_FilesOrganized"
    folders_organized = desktop / "_FoldersOrganized"
    screenshots_folder = desktop / "_Screenshots"
    
    # Organize files by type
    if files_by_type:
        files_organized.mkdir(exist_ok=True)
        print("\n📄 Organizing files by type...")
        
        for category, files in files_by_type.items():
            category_folder = files_organized / category
            category_folder.mkdir(exist_ok=True)
            
            for file in files:
                try:
                    destination = category_folder / file.name
                    if destination.exists():
                        n = 1
                        while destination.exists():
                            destination = category_folder / f"{file.stem}_{n}{file.suffix}"
                            n += 1
                    
                    shutil.move(str(file), str(destination))
                    print(f"  ✓ {file.name} → _FilesOrganized/{category}/")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ Error moving {file.name}: {e}")
    
    # Organize screenshots
    if screenshots:
        screenshots_folder.mkdir(exist_ok=True)
        print("\n📸 Organizing screenshots...")
        
        # Create date-based subfolders for screenshots
        for screenshot in screenshots:
            try:
                # Get date from file
                mod_time = datetime.fromtimestamp(screenshot.stat().st_mtime)
                date_folder = screenshots_folder / mod_time.strftime("%Y-%m")
                date_folder.mkdir(exist_ok=True)
                
                destination = date_folder / screenshot.name
                shutil.move(str(screenshot), str(destination))
                print(f"  ✓ {screenshot.name} → _Screenshots/{mod_time.strftime('%Y-%m')}/")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Error moving {screenshot.name}: {e}")
    
    # Organize folders by category
    if folders_by_category:
        folders_organized.mkdir(exist_ok=True)
        print("\n📁 Organizing folders by category...")
        
        for category, folders in folders_by_category.items():
            category_folder = folders_organized / category
            category_folder.mkdir(exist_ok=True)
            
            for folder in folders:
                try:
                    destination = category_folder / folder.name
                    shutil.move(str(folder), str(destination))
                    print(f"  ✓ {folder.name}/ → _FoldersOrganized/{category}/")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ Error moving {folder.name}: {e}")
    
    return moved_count

def organize_by_date():
    """Organize desktop items by date"""
    desktop = Path.home() / "Desktop"
    by_date_folder = desktop / "_OrganizedByDate"
    by_date_folder.mkdir(exist_ok=True)
    
    moved_count = 0
    
    print("\n📅 Organizing by date...")
    
    for item in desktop.iterdir():
        # Skip hidden and already organized
        if item.name.startswith('.') or item.name.startswith('_'):
            continue
        
        try:
            # Get modification date
            mod_time = datetime.fromtimestamp(item.stat().st_mtime)
            
            # Create year/month folder
            date_folder = by_date_folder / mod_time.strftime("%Y") / mod_time.strftime("%m-%B")
            date_folder.mkdir(parents=True, exist_ok=True)
            
            # Move item
            destination = date_folder / item.name
            shutil.move(str(item), str(destination))
            print(f"  ✓ {item.name} → {mod_time.strftime('%Y/%m-%B')}/")
            moved_count += 1
            
        except Exception as e:
            print(f"  ✗ Error moving {item.name}: {e}")
    
    return moved_count

def main():
    """Main program"""
    print("""
🧹 DESKTOP DEEP ORGANIZER
=========================

Make your Desktop spotless with intelligent organization!
""")
    
    # Analyze desktop
    files_by_type, folders_by_category, screenshots = analyze_desktop()
    
    if not any([files_by_type, folders_by_category, screenshots]):
        print("\n✨ Your Desktop is already clean!")
        return
    
    # Show options
    choice = show_organization_options()
    
    moved_count = 0
    
    if choice == '1':
        # Smart organize
        moved_count = smart_organize(files_by_type, folders_by_category, screenshots)
        
    elif choice == '2':
        # By type only
        moved_count = smart_organize(files_by_type, {}, [])
        
    elif choice == '3':
        # By date
        moved_count = organize_by_date()
        
    elif choice == '4':
        # Screenshots only
        moved_count = smart_organize({}, {}, screenshots)
        
    elif choice == '5':
        # Deep clean - organize everything
        print("\n🧹 Deep Clean mode - organizing EVERYTHING...")
        response = input("⚠️  This will organize ALL files and folders. Continue? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            moved_count = smart_organize(files_by_type, folders_by_category, screenshots)
        
    elif choice == '6':
        # Custom - let user choose
        print("\n📝 What to organize?")
        print("1. Files only")
        print("2. Folders only")
        print("3. Screenshots only")
        print("4. Files + Screenshots")
        custom = input("Choice: ").strip()
        
        if custom == '1':
            moved_count = smart_organize(files_by_type, {}, [])
        elif custom == '2':
            moved_count = smart_organize({}, folders_by_category, [])
        elif custom == '3':
            moved_count = smart_organize({}, {}, screenshots)
        elif custom == '4':
            moved_count = smart_organize(files_by_type, {}, screenshots)
    
    else:
        print("\n👌 Cancelled!")
        return
    
    # Summary
    if moved_count > 0:
        print(f"\n✅ Success! Organized {moved_count} items")
        print("\n🎯 Your Desktop now has these organization folders:")
        desktop = Path.home() / "Desktop"
        for folder in desktop.iterdir():
            if folder.is_dir() and folder.name.startswith('_'):
                print(f"  {folder.name}/")
        
        print("\n💡 TIP: You can move items out of these folders anytime!")

if __name__ == "__main__":
    try:
        main()
        input("\n⏸️  Press Enter to close...")
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\n⏸️  Press Enter to close...")