script_folder="/Users/michalmaj/Documents/cpp/improc++/cmake-build-release/conan/build/Release/generators"
echo "echo Restoring environment" > "$script_folder/deactivate_conanrunenv-release-armv8.sh"
for v in OPENSSL_MODULES
do
    is_defined="true"
    value=$(printenv $v) || is_defined="" || true
    if [ -n "$value" ] || [ -n "$is_defined" ]
    then
        echo export "$v='$value'" >> "$script_folder/deactivate_conanrunenv-release-armv8.sh"
    else
        echo unset $v >> "$script_folder/deactivate_conanrunenv-release-armv8.sh"
    fi
done


export OPENSSL_MODULES="/Users/michalmaj/.conan2/p/b/opens9bf56e2ebc65d/p/lib/ossl-modules"