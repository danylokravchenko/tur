for i in 1 2 3 4; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"qwen-hybrid\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Tell me a fact number $i\"}]
    }" &
done
wait
