import { useMemo, useRef, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

const MIN_LAMBDA = 0
const MAX_LAMBDA = 3

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function App() {
  const apiBase =
    (import.meta as ImportMeta).env.VITE_API_URL ?? "http://localhost:8000"
  const padRef = useRef<HTMLDivElement | null>(null)

  const [userId, setUserId] = useState("demo-user")
  const [prompt, setPrompt] = useState(
    "Write a short introduction for a research paper on context steering."
  )
  const [styleA, setStyleA] = useState("Use a formal academic tone.")
  const [styleB, setStyleB] = useState("Be concise and direct.")
  const [lambdaA, setLambdaA] = useState(1.0)
  const [lambdaB, setLambdaB] = useState(1.0)
  const [draft, setDraft] = useState("")
  const [inferred, setInferred] = useState<{ a: number; b: number } | null>(
    null
  )
  const [isGenerating, setIsGenerating] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [status, setStatus] = useState<string | null>(null)

  const handlePadUpdate = (clientX: number, clientY: number) => {
    if (!padRef.current) return
    const rect = padRef.current.getBoundingClientRect()
    const x = clamp(clientX - rect.left, 0, rect.width)
    const y = clamp(clientY - rect.top, 0, rect.height)
    const nextA =
      MIN_LAMBDA + (x / rect.width) * (MAX_LAMBDA - MIN_LAMBDA)
    const nextB =
      MIN_LAMBDA + ((rect.height - y) / rect.height) * (MAX_LAMBDA - MIN_LAMBDA)
    setLambdaA(Number(nextA.toFixed(2)))
    setLambdaB(Number(nextB.toFixed(2)))
  }

  const handleGenerate = async () => {
    setIsGenerating(true)
    setStatus(null)
    try {
      const res = await fetch(`${apiBase}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          prompt,
          style_a: styleA,
          style_b: styleB,
          lambda_a: lambdaA,
          lambda_b: lambdaB,
        }),
      })
      if (!res.ok) {
        throw new Error(`Generate failed: ${res.status}`)
      }
      const data = await res.json()
      setDraft(data.draft_text ?? "")
      setStatus("Draft generated.")
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Generate failed.")
    } finally {
      setIsGenerating(false)
    }
  }

  const handleFeedback = async () => {
    if (!draft.trim()) {
      setStatus("Draft is empty.")
      return
    }
    setIsSaving(true)
    setStatus(null)
    try {
      const res = await fetch(`${apiBase}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          prompt,
          style_a: styleA,
          style_b: styleB,
          edited_text: draft,
        }),
      })
      if (!res.ok) {
        throw new Error(`Feedback failed: ${res.status}`)
      }
      const data = await res.json()
      const next = {
        a: Number(data.inferred_lambda_a ?? lambdaA),
        b: Number(data.inferred_lambda_b ?? lambdaB),
      }
      setInferred(next)
      setLambdaA(next.a)
      setLambdaB(next.b)
      setStatus("Feedback saved. Updated steering point.")
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Feedback failed.")
    } finally {
      setIsSaving(false)
    }
  }

  const handlePosition = useMemo(() => {
    const ratioA = (lambdaA - MIN_LAMBDA) / (MAX_LAMBDA - MIN_LAMBDA)
    const ratioB = (lambdaB - MIN_LAMBDA) / (MAX_LAMBDA - MIN_LAMBDA)
    return {
      left: `${clamp(ratioA, 0, 1) * 100}%`,
      top: `${(1 - clamp(ratioB, 0, 1)) * 100}%`,
    }
  }, [lambdaA, lambdaB])

  return (
    <div className="min-h-screen bg-background px-6 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold">Context Steering POC</h1>
          <p className="text-sm text-muted-foreground">
            Steer writing with two style axes, then edit to feed back into
            steering.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.05fr_1fr]">
          <Card>
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="grid gap-2">
                <label className="text-sm font-medium">User ID</label>
                <Input value={userId} onChange={(e) => setUserId(e.target.value)} />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium">Prompt</label>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium">Style A</label>
                <Input value={styleA} onChange={(e) => setStyleA(e.target.value)} />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium">Style B</label>
                <Input value={styleB} onChange={(e) => setStyleB(e.target.value)} />
              </div>

              <div className="grid gap-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">2D Style Pad</label>
                  <div className="flex gap-2">
                    <Badge>λA {lambdaA.toFixed(2)}</Badge>
                    <Badge>λB {lambdaB.toFixed(2)}</Badge>
                  </div>
                </div>
                <div
                  ref={padRef}
                  className="relative h-48 w-full rounded-lg border border-border bg-muted/30"
                  onPointerDown={(event) => {
                    handlePadUpdate(event.clientX, event.clientY)
                    ;(event.currentTarget as HTMLDivElement).setPointerCapture(
                      event.pointerId
                    )
                  }}
                  onPointerMove={(event) => {
                    if (event.buttons === 1) {
                      handlePadUpdate(event.clientX, event.clientY)
                    }
                  }}
                >
                  <div
                    className="absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary shadow"
                    style={handlePosition}
                  />
                  <div className="pointer-events-none absolute inset-0 flex items-end justify-between px-2 pb-2 text-xs text-muted-foreground">
                    <span>Low A</span>
                    <span>High A</span>
                  </div>
                  <div className="pointer-events-none absolute inset-0 flex flex-col items-start justify-between px-2 py-2 text-xs text-muted-foreground">
                    <span>High B</span>
                    <span>Low B</span>
                  </div>
                </div>
              </div>

              <Button onClick={handleGenerate} disabled={isGenerating}>
                {isGenerating ? "Generating..." : "Generate"}
              </Button>
              {status && <p className="text-sm text-muted-foreground">{status}</p>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Draft Output</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <Textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                className="min-h-[320px]"
              />
              <div className="flex items-center justify-between">
                <Button variant="secondary" onClick={handleFeedback} disabled={isSaving}>
                  {isSaving ? "Saving..." : "Save Feedback"}
                </Button>
                {inferred && (
                  <div className="text-xs text-muted-foreground">
                    Inferred λA {inferred.a.toFixed(2)}, λB {inferred.b.toFixed(2)}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default App
