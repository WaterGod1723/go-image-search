export PATH="$PATH:$(go env GOPATH)/bin"; export GOSUMDB=off; wails build -tags wails 2>&1 | tail -40; echo "EXIT: ${PIPESTATUS[0]}"
