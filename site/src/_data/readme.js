// The landing page's commands come from README.md, so its pinned revision and flags stay in one place.
import { readRepoFile } from "../../lib/book.js";

const blocks = [...readRepoFile("README.md").matchAll(/```sh\n([\s\S]*?)```/g)].map((match) => match[1].trim());
if (blocks.length < 4) throw new Error("README.md: expected install, run, serve and connect commands");

const [install, run, serve, connect] = blocks;

export default { install, run, serve, connect };
