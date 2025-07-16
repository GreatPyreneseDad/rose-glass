#!/usr/bin/env python3
"""
Folder Browser & Organizer
Browse to any folder and organize its contents by file type
"""

import os
import shutil
from pathlib import Path

# File categories
FILE_CATEGORIES = {
    'Images': ['.jpg', '.jpeg', '.png', '.gif', '.heic', '.bmp', '.svg', '.webp', '.ico', '.tiff'],
    'Documents': ['.pdf', '.doc', '.docx', '.txt', '.pages', '.odt', '.rtf', '.tex'],
    'Spreadsheets': ['.xls', '.xlsx', '.numbers', '.csv', '.ods'],
    'Presentations': ['.ppt', '.pptx', '.key', '.odp'],
    'Videos': ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm'],
    'Audio': ['.mp3', '.m4a', '.wav', '.aac', '.flac', '.aiff', '.ogg', '.wma'],
    'Archives': ['.zip', '.dmg', '.tar', '.gz', '.rar', '.7z', '.pkg', '.deb'],
    'Code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.sh', '.c', '.php', '.rb', '.go'],
    'Design': ['.psd', '.ai', '.sketch', '.fig', '.xd', '.indd'],
    'Data': ['.json', '.xml', '.sql', '.db', '.sqlite'],
    'Ebooks': ['.epub', '.mobi', '.azw', '.azw3', '.fb2'],
    'Fonts': ['.ttf', '.otf', '.woff', '.woff2', '.eot'],
}

def show_current_location(path):
    """Show current location and contents"""
    print(f"\n📍 Current location: {path}")
    print("=" * 60)
    
    # List directories first
    dirs = []
    files = []
    
    try:
        for item in path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dirs.append(item)
            elif item.is_file() and not item.name.startswith('.'):
                files.append(item)
    except PermissionError:
        print("❌ Permission denied for this folder")
        return False
    
    # Show folders
    if dirs:
        print("\n📁 FOLDERS:")
        for i, d in enumerate(sorted(dirs)[:20], 1):  # Show max 20
            print(f"  {i}. {d.name}/")
        if len(dirs) > 20:
            print(f"  ... and {len(dirs) - 20} more folders")
    
    # Show file summary
    if files:
        # Count by type
        file_types = {}
        for f in files:
            ext = f.suffix.lower()
            category = "Other"
            for cat, extensions in FILE_CATEGORIES.items():
                if ext in extensions:
                    category = cat
                    break
            file_types[category] = file_types.get(category, 0) + 1
        
        print(f"\n📄 FILES: {len(files)} total")
        for cat, count in sorted(file_types.items()):
            print(f"  {cat}: {count} files")
    else:
        print("\n📄 No files in this folder")
    
    return True

def browse_to_folder():
    """Interactive folder browser"""
    current_path = Path.home()  # Start at home directory
    
    while True:
        # Show current location
        if not show_current_location(current_path):
            # Go back if permission denied
            current_path = current_path.parent
            continue
        
        # Show options
        print("\n🎯 OPTIONS:")
        print("  [number]  - Enter a folder")
        print("  ..        - Go up one level")
        print("  ~         - Go to home directory")
        print("  organize  - Organize THIS folder")
        print("  path      - Type a specific path")
        print("  quit      - Exit program")
        
        choice = input("\n👉 Your choice: ").strip().lower()
        
        if choice == 'quit':
            return None
        elif choice == 'organize':
            return current_path
        elif choice == '..':
            current_path = current_path.parent
        elif choice == '~':
            current_path = Path.home()
        elif choice == 'path':
            custom = input("Enter path: ").strip()
            try:
                new_path = Path(custom).expanduser()
                if new_path.exists() and new_path.is_dir():
                    current_path = new_path
                else:
                    print("❌ Invalid path!")
            except:
                print("❌ Error with that path!")
        elif choice.isdigit():
            # Navigate to numbered folder
            dirs = [d for d in current_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
            dirs = sorted(dirs)[:20]  # Only first 20
            
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                current_path = dirs[idx]
            else:
                print("❌ Invalid number!")

def organize_folder(folder_path):
    """Organize the selected folder"""
    print(f"\n🎯 Organizing: {folder_path}")
    print("=" * 60)
    
    # Scan files
    files_to_move = []
    file_count = {}
    
    for item in folder_path.iterdir():
        if item.is_file() and not item.name.startswith('.'):
            ext = item.suffix.lower()
            
            # Find category
            category = "Other"
            for cat, extensions in FILE_CATEGORIES.items():
                if ext in extensions:
                    category = cat
                    break
            
            files_to_move.append((item, category))
            file_count[category] = file_count.get(category, 0) + 1
    
    if not files_to_move:
        print("✅ No files to organize!")
        return
    
    # Show what will happen
    print(f"\n📊 Found {len(files_to_move)} files to organize:")
    for cat, count in sorted(file_count.items()):
        print(f"  {cat}: {count} files")
    
    # Confirm
    response = input("\n🤔 Proceed with organization? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("👌 Cancelled!")
        return
    
    # Organize
    print("\n🚀 Organizing...")
    moved = 0
    errors = 0
    
    for file_path, category in files_to_move:
        # Create subfolder
        subfolder = folder_path / category
        subfolder.mkdir(exist_ok=True)
        
        try:
            destination = subfolder / file_path.name
            
            # Handle duplicates
            if destination.exists():
                n = 1
                while destination.exists():
                    name = f"{file_path.stem}_{n}{file_path.suffix}"
                    destination = subfolder / name
                    n += 1
            
            shutil.move(str(file_path), str(destination))
            print(f"  ✓ {file_path.name} → {category}/")
            moved += 1
            
        except Exception as e:
            print(f"  ✗ Error with {file_path.name}: {e}")
            errors += 1
    
    # Summary
    print(f"\n✅ Complete! Moved {moved} files")
    if errors:
        print(f"⚠️  {errors} errors occurred")
    
    # Show result
    print(f"\n📁 Your organized folder now has these subfolders:")
    for subfolder in sorted(folder_path.iterdir()):
        if subfolder.is_dir() and subfolder.name in FILE_CATEGORIES:
            count = len(list(subfolder.iterdir()))
            print(f"  {subfolder.name}/  ({count} files)")

def main():
    """Main program"""
    print("""
🗂️  FOLDER BROWSER & ORGANIZER
==============================

Browse to any folder and organize its contents into subfolders by type!
""")
    
    while True:
        # Browse to folder
        selected_folder = browse_to_folder()
        
        if selected_folder is None:
            print("\n👋 Goodbye!")
            break
        
        # Organize it
        organize_folder(selected_folder)
        
        # Ask if they want to organize another
        print("\n" + "="*60)
        again = input("\n🔄 Organize another folder? (yes/no): ").lower()
        if again not in ['yes', 'y']:
            print("\n👋 Thanks for using Folder Organizer!")
            break

if __name__ == "__main__":
    try:
        main()
        input("\n⏸️  Press Enter to close...")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\n⏸️  Press Enter to close...")