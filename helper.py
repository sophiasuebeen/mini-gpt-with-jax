import csv
from pathlib import Path


def load_stories_from_file(file_path, *, text_column="text", limit=None, max_stories=None):
    """Load TinyStories from a .txt file or a Kaggle-style CSV file."""
    path = Path(file_path)
    if limit is None:
        limit = max_stories

    if path.suffix == ".csv":
        stories = []
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(
                    f"Expected a CSV column named {text_column!r}; "
                    f"found {reader.fieldnames!r}."
                )

            for row in reader:
                story = row[text_column].strip()
                if story:
                    stories.append(story)
                    if limit is not None and len(stories) >= limit:
                        break
        return stories

    data = path.read_text(encoding="utf-8", errors="replace")
    stories = [story.strip() for story in data.split("<|endoftext|>") if story.strip()]
    return stories if limit is None else stories[:limit]
