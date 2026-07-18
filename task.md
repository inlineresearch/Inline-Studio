# Creator task: build a 20-second story using Inline Studio

Use Inline Studio to plan, generate, and curate the shots of a short story, then share us your experience about Inline Studio.
This is a real production exercise. We want to see Inline Studio used end to end on an actual story, and learn whether it genuinely helps you plan, design and execute a production pipeline, and what we should build next.

## The brief

Make a roughly 20-second video story with multiple shots (4 to 8 is a good range) and audio (music, voice, or sound effects). Keep the story simple. The point is the workflow, not the runtime, and the creative direction is entirely yours.

## Setup

Inline Studio runs on your machine and uses your own ComfyUI for generation. Full setup lives in the README:

- Install and run the app: see [README, Getting started](README.md#getting-started).
- Connect ComfyUI (local GPU or cloud): see [README, Bring your own ComfyUI](README.md#bring-your-own-comfyui).
- Cloud (RunPod) walkthrough video: https://youtu.be/JovhfHhxqdM?si=lHQo9qzR_fCZwYCL

Quick reminder: start ComfyUI with CORS enabled (`python main.py --enable-cors-header`), then paste its URL (for local, usually `http://127.0.0.1:8188`) into the Generate tab.

No local GPU and need a cloud machine to run ComfyUI? If you need access to RunPod (or help covering the cost), let us know and we'll sort it out before you start.

## Suggested flow

This is a guide, not a rulebook. Work however feels natural and note where it does or does not fit your process.

1. Storyboard your shots as frame nodes on the canvas, one per shot.
2. Bring in any reference assets (images, clips, audio) and drop them onto frames as inputs.
3. In the Generate tab, link a ComfyUI workflow to a shot, then generate.
4. Generate a few takes per shot and star the hero (the take you want to keep).
5. Chain shots where it helps: wire a Preview's output into the next shot's input so a look or character carries forward.
6. Bring audio in as a reference for timing and mood.
7. Export the shots folder from Inline Studio (numbered hero takes, in order).
8. Combine the shots and add audio in your editor of choice (DaVinci Resolve, Premiere, CapCut, etc.) to render the final 20-second video.

Note: Inline Studio does not stitch a combined video or mix audio yet, so step 8 is external by design. How that hand-off feels is exactly the kind of thing we want to hear about.

## Deliverables

1. A handful of screenshots that show your process and flow (the canvas/storyboard, a generate step, chaining shots, and the exported shots).
2. The final combined 20-second video (assembled in your external editor).
3. A short feedback write-up in a private Google Doc, using the template below.

## Feedback template

Copy this into your Google Doc and fill it in. Be candid. Negative feedback is just as useful as positive.

```
Inline Studio creator task feedback

Name / handle:
Date:
Story (one line):
Links: [screenshots] [final video]

1. What did you like? What worked well?

2. What should we improve? What got in your way?

3. What other features would make Inline Studio more useful to you?

4. Did Inline Studio genuinely help you plan or design your production pipeline? Why or why not?

5. Where did it slow you down, or where did you drop out to other tools (and which tools)?

6. The core model is "a shot is a slot with a history of takes." Did that fit how
   you think about your work? Where did it help or get in the way?

7. Overall: would you use this again on a real project? (1 to 5) and one sentence on why.

Anything else you want us to know:
```

## Timeline

2 days from when you start.

## Submission

Put your write-up, screenshots, and the final video (or links to them) in a single private Google Doc, and share it with: `team@inlinestudio.art`.
