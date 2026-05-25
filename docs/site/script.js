const docsRoot = "../";
const githubRepoOwner = "JayNightmare";
const githubRepoName = "FakeNews";
const githubRepoRef = "main";
const ignoredDocsPrefixes = ["site/"];

const fileListEl = document.getElementById("fileList");
const contentEl = document.getElementById("content");
const titleEl = document.getElementById("currentFileTitle");
const loaderEl = document.getElementById("loader");

let availableFiles = [];

function normalizeFilePath(file) {
	return file.replace(/^\.\//, "").replace(/^\//, "");
}

function buildDocUrl(file = "") {
	return new URL(
		normalizeFilePath(file),
		new URL(docsRoot, window.location.href),
	).toString();
}

function normalizeDirectoryPath(dir = "") {
	const normalized = normalizeFilePath(dir);
	if (!normalized) {
		return "";
	}
	return normalized.endsWith("/") ? normalized : `${normalized}/`;
}

function uniqueFiles(files) {
	return [...new Set(files.map(normalizeFilePath))];
}

function shouldIgnoreDocPath(file) {
	const normalized = normalizeFilePath(file);
	return ignoredDocsPrefixes.some((prefix) =>
		normalized.startsWith(prefix),
	);
}

function isLocalDocsHost() {
	return ["127.0.0.1", "localhost"].includes(window.location.hostname);
}

function formatTitleSegment(segment) {
	return segment
		.replace(/\.md$/i, "")
		.replace(/[_-]+/g, " ")
		.replace(/\s+/g, " ")
		.replace(/^\s+|\s+$/g, "")
		.trim()
		.replace(/\b\w/g, (char) => char.toUpperCase());
}

function getDisplayName(file) {
	const segments = normalizeFilePath(file)
		.split("/")
		.map(formatTitleSegment);
	return segments.join(" / ");
}

async function getDocNamesFromGitHub() {
	const apiUrl = `https://api.github.com/repos/${githubRepoOwner}/${githubRepoName}/git/trees/${githubRepoRef}?recursive=1`;
	const response = await fetch(apiUrl, {
		headers: {
			Accept: "application/vnd.github+json",
		},
	});
	if (!response.ok) {
		throw new Error(
			`GitHub API request failed with ${response.status}`,
		);
	}

	const payload = await response.json();
	if (!Array.isArray(payload.tree)) {
		throw new Error("GitHub API did not return a repository tree.");
	}

	return uniqueFiles(
		payload.tree
			.filter((entry) => entry.type === "blob")
			.map((entry) => entry.path)
			.filter((path) => path.startsWith("docs/"))
			.map((path) => path.slice("docs/".length))
			.filter((path) => path.endsWith(".md"))
			.filter((path) => !shouldIgnoreDocPath(path)),
	);
}

async function getDocNamesFromDirectoryListing() {
	const files = [];
	const queuedDirs = [""];
	const seenDirs = new Set(queuedDirs);

	while (queuedDirs.length) {
		const dir = queuedDirs.shift() || "";
		const response = await fetch(buildDocUrl(dir));
		if (!response.ok) {
			throw new Error(
				`Directory request failed with ${response.status}`,
			);
		}

		const text = await response.text();
		const parser = new DOMParser();
		const doc = parser.parseFromString(text, "text/html");
		const links = doc.querySelectorAll("a");

		links.forEach((link) => {
			const href = link.getAttribute("href");
			if (!href || href === "../" || href === "./") {
				return;
			}

			if (href.endsWith("/")) {
				const nextDir = normalizeDirectoryPath(
					`${dir}${href}`,
				);
				if (
					!nextDir ||
					shouldIgnoreDocPath(nextDir) ||
					seenDirs.has(nextDir)
				) {
					return;
				}
				seenDirs.add(nextDir);
				queuedDirs.push(nextDir);
				return;
			}

			if (href.endsWith(".md")) {
				const file = normalizeFilePath(`${dir}${href}`);
				if (!shouldIgnoreDocPath(file)) {
					files.push(file);
				}
			}
		});
	}

	return uniqueFiles(files);
}

// Gets the names of all markdown files in the docs/ folder.
async function getDocNames() {
	const strategies = isLocalDocsHost()
		? [getDocNamesFromDirectoryListing, getDocNamesFromGitHub]
		: [getDocNamesFromGitHub, getDocNamesFromDirectoryListing];

	for (const strategy of strategies) {
		try {
			const files = await strategy();
			if (files.length) {
				return files;
			}
		} catch (err) {
			console.warn(
				"Could not discover docs with strategy.",
				err,
			);
		}
	}

	return [];
}

// Gets the content of a markdown file from the docs/ folder.
async function getDocContent(docName) {
	const response = await fetch(encodeURI(buildDocUrl(docName)));
	if (!response.ok) {
		return null;
	}
	return response.text();
}

// Displays the content of a markdown file in the console.
async function displayDocContent(docName) {
	const content = await getDocContent(docName);
	if (content) {
		console.log(content);
	} else {
		console.log(`Document "${docName}" not found.`);
	}
}

// Configure marked to use highlight.js for code blocks
marked.setOptions({
	highlight: function (code, lang) {
		if (lang && hljs.getLanguage(lang)) {
			return hljs.highlight(code, { language: lang }).value;
		}
		return hljs.highlightAuto(code).value;
	},
	breaks: true,
});

// Initialize sidebar
async function initSidebar() {
	availableFiles = await getDocNames();
	fileListEl.innerHTML = "";

	availableFiles.forEach((file) => {
		const li = document.createElement("li");
		li.className = "file-item";
		const displayName = getDisplayName(file);
		li.textContent = displayName;
		li.onclick = () => loadFile(file, li, displayName);
		fileListEl.appendChild(li);
	});

	if (!availableFiles.length) {
		fileListEl.innerHTML =
			'<li class="file-item">No markdown files found</li>';
		return;
	}

	// Check URL for specific file
	const urlParams = new URLSearchParams(window.location.search);
	const fileToLoad = urlParams.get("file");

	if (fileToLoad && availableFiles.includes(fileToLoad)) {
		const itemIndex = availableFiles.indexOf(fileToLoad);
		const li = fileListEl.children[itemIndex];
		li.click();
	} else if (availableFiles.length > 0) {
		// Load first file by default
		fileListEl.children[0].click();
	}
}

// Load and render markdown file
async function loadFile(filename, element, displayName) {
	// Update UI state
	document.querySelectorAll(".file-item").forEach((el) =>
		el.classList.remove("active"),
	);
	if (element) element.classList.add("active");
	titleEl.textContent = displayName;

	// Update URL without reloading
	const newUrl = new URL(window.location);
	newUrl.searchParams.set("file", filename);
	window.history.pushState({}, "", newUrl);

	// Hide mobile menu if open
	document.getElementById("fileList").classList.remove("open");

	// Fetch and render
	contentEl.style.opacity = "0.5";
	loaderEl.style.display = "block";

	try {
		const response = await fetch(encodeURI(buildDocUrl(filename)));
		if (!response.ok) throw new Error("File not found");
		const text = await response.text();

		// Render markdown
		contentEl.innerHTML = marked.parse(text);

		// Scroll to top
		contentEl.scrollTop = 0;
	} catch (err) {
		contentEl.innerHTML = `
          <div style="color: var(--danger); text-align: center; margin-top: 100px;">
            <h3>Failed to load document</h3>
            <p>${err.message}</p>
            <p style="font-size: 14px; opacity: 0.7;">Make sure you are running this through a web server (or GitHub Pages), as local file:// requests for fetch() might be blocked by CORS.</p>
          </div>
        `;
	} finally {
		contentEl.style.opacity = "1";
		loaderEl.style.display = "none";
	}
}

// Run on load
initSidebar();

// Handle back/forward browser buttons
window.addEventListener("popstate", () => {
	const urlParams = new URLSearchParams(window.location.search);
	const file = urlParams.get("file");
	if (file && availableFiles.includes(file)) {
		const itemIndex = availableFiles.indexOf(file);
		const li = fileListEl.children[itemIndex];
		const displayName = li.textContent;
		loadFile(file, li, displayName);
	}
});
